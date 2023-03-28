# add drop out layer and make the model can be scaled up
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
# cd to the path of current file
os.chdir(current_path)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_tensor_type(torch.FloatTensor)



# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 400
learning_rate = 3e-4
eval_iters = 200
max_new_tokens = 500

# add embedding layer and position embedding layer
embedding_size = 384
head_size = 64
num_heads = 6
num_layers = 6
dropout = 0.2
num_epochs = 100
# ---------------------------


torch.manual_seed(1337)

# load text dataset
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
# list all the chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from char to index
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
token_encoder = lambda s: [stoi[c] for c in s]
token_decoder = lambda l: ''.join([itos[i] for i in l])

# train and test data





    
# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

class LayerNorm1d:
    def __init__(self, dim, eps=1e-5):
        # dim: number of features
        self.eps = eps
        self.gamma = torch.ones(dim) # layer norm gain
        self.beta = torch.zeros(dim) # layer norm bias
    
    def __call__(self, x):
        # x: (batch_size, block_size, embed_size)
        xmean = x.mean(dim=-1, keepdim=True) # layer mean (across all features)
        xvar = x.var(dim=-1, keepdim=True) # layer variance
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize
        self.out = self.gamma * xhat + self.beta # scale and shift
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


    

class BatchNorm1d:
    
    def __init__(self, dim, eps=1e-5, momentum=0.1, training=True):
        # dim: number of features
        self.eps = eps
        self.momentum = momentum
        self.training = training
        self.gamma = torch.ones(dim) # layer norm gain
        self.beta = torch.zeros(dim) # layer norm bias
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        # x: (batch_size, block_size, embed_size)
        if self.training:
            xmean = x.mean(dim=0, keepdim=True) # layer mean (across all features)
            xvar = x.var(dim=0, keepdim=True) # layer variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize
        self.out = self.gamma * xhat + self.beta # scale and shift
        
        # update running mean and variance
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_size, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, block_size, embed_size) -> (batch_size, block_size, num_heads * head_size)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size, embed_size, block_size, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, block_size, embed_size)
        k = self.key(x) # (batch_size, block_size, head_size)
        q = self.query(x) # (batch_size, block_size, head_size)
        # compute attention scores
        wei = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5) # (batch_size, block_size, block_size)
        wei = wei.masked_fill(self.mask[:self.block_size, :self.block_size] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (batch_size, block_size, block_size)
        wei = self.dropout(wei)
        
        v = self.value(x) # (batch_size, block_size, head_size)
        out = torch.matmul(wei, v) # (batch_size, block_size, head_size)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, input_size), # projection
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self, embedding_size, head_size, block_size, num_heads, dropout=0.1):
        super().__init__()
        # input: (batch_size, block_size, embedding_size), output: (batch_size, block_size, head_size * num_heads)
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size, embed_size=embedding_size, block_size=block_size)
        # input: (batch_size, block_size, head_size * num_heads), output: (batch_size, block_size, head_size * num_heads)
        self.ffwd = FeedForward(input_size=embedding_size, output_size=4*num_heads*head_size, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # need to make sure that the dimension of x and self.sa(x) are the same
        x = x + self.ffwd(self.ln2(x)) # need to make sure that the dimension of x and self.ffwd(x) are the same
        return x


# bigram model
class BigramModel(nn.Module):
    def __init__(self, embedding_size, head_size, vocab_size, block_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        multihead_size = head_size * num_heads
        self.token_embedding_tabel = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_tabel = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, head_size, block_size, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(multihead_size, vocab_size)
        
            
    def forward(self, inputs, targets=None):
        # inputs: (batch_size, block_size)
        # targets: (batch_size, block_size)
        batch_size, block_size = inputs.shape
        tok_emb = self.token_embedding_tabel(inputs) # (batch_size, block_size, embedding_size) / (B, T, C)
        pos_emb = self.position_embedding_tabel(torch.arange(block_size)) # (block_size, embedding_size) / (T, C) [same for all batch]
        x = tok_emb + pos_emb # (batch_size, block_size, embedding_size) / (B, T, C)
        x = self.blocks(x) # (batch_size, block_size, head_size) / (B, T, C)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_rs = logits.view(B*T, C)
            targets_rs = targets.view(B*T)
            # loss = F.cross_entropy(logits.transpose(1,2), targets)
            loss = F.cross_entropy(logits_rs, targets_rs)
        return logits, loss
    
    def generate(self, idx, max_new_token):
        # idx: (batch_size, block_size) array of indices in the current context
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:] # (batch_size, block_size)
            logits, loss = self(idx_cond) # (batch_size, block_size, vocab_size)
            # only focus on the last token
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1) # (batch_size, vocab_size)
            idx_next = torch.multinomial(input=probs, num_samples=1) # sampling: (batch_size, 1)
            idx = torch.cat([idx, idx_next], dim=1) # (batch_size, block_size+1)
        
        return idx



# creat dataset for train and val data
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        # self.data = torch.tensor(data)
        self.data = data
        self.block_size = block_size
        self.num_blocks = len(self.data) // self.block_size
    
    def __len__(self):
        return self.num_blocks
    
    def __getitem__(self, index):
        x = self.data[index * self.block_size: (index + 1) * self.block_size]
        y = self.data[index * self.block_size + 1: (index + 1) * self.block_size + 1]
        return x, y
    


# ---------------------------
def init_process(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)



def training(rank, world_size):
    init_process(rank, world_size)
    
    data = torch.tensor(token_encoder(text), dtype=torch.long)
    len_train = int(0.9 * len(data))
    train_data = data[:len_train]
    val_data = data[len_train:]
    
    train_set = TextDataset(train_data, block_size)
    val_set = TextDataset(val_data, block_size)
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler)
    
    model = BigramModel(embedding_size, head_size, vocab_size, block_size, num_heads, num_layers, dropout).to(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(rank)
            yb = yb.to(rank)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Rank: {rank},  epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}")
                
            if batch_idx % 100 == 0:
                train_loss, val_loss = evaluate(rank, model, val_loader)
                print(f"Rank: {rank},  epoch: {epoch}, batch: {batch_idx}, train_loss: {train_loss}, val_loss: {val_loss}")
    
    context = torch.zeros((1,block_size), dtype=torch.long)
    print(token_decoder(model.generate(context, max_new_tokens)[0].tolist()))

            

def evaluate(rank, model, train_loader, val_loader):
    model.eval()
    train_loss = 0
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(rank)
            yb = yb.to(rank)
            logits, loss = model(xb, yb)
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        for batch_idx, (xb, yb) in enumerate(val_loader):
            xb = xb.to(rank)
            yb = yb.to(rank)
            logits, loss = model(xb, yb)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        
    model.train()
    return train_loss, val_loss

# --------------------------- 
if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    world_size = 2  # Number of processes
    
    # mp.spawn(training, args=(world_size,), nprocs=world_size, join=True)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=training, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()








    
