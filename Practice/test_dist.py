import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


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
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size, embed_size=embedding_size, block_size=block_size, dropout=dropout)
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
        self.block_size = block_size
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
            idx_cond = idx[:, -self.block_size:] # (batch_size, block_size)
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
    

def init_process(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    # set a random port number
    port_number = 29000
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port_number)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

# ---------------------------------------------------------------


# ---------------------------------------------------------------
def train(rank, world_size, model, train_set, model_paras):
    
    init_process(rank, world_size)
    model = model.to(torch.device(f'cpu:{rank}'))
    model = DDP(model)
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=model_paras['batch_size'], sampler=train_sampler, num_workers=3)
    num_batches = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_paras['lr'])
    
    # print(f"Starting training process on CPU: {rank}, num_batches: {num_batches}")
    
    
    model.train()
    for epoch in range(model_paras['num_epochs']):
        train_sampler.set_epoch(epoch)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(torch.device(f'cpu:{rank}'))
            targets = targets.to(torch.device(f'cpu:{rank}'))
            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            
            # if batch_idx % 10 == 0:
            print(f"Rank: {rank},  epoch: {epoch}, batch: {batch_idx} / {num_batches}, loss: {loss.item()}")
                



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


# model_paras = {'embedding_size': 384,
#                'head_size': 64,
#                'vocab_size': vocab_size,
#                'block_size': 256,
#                'num_heads': 6,
#                'num_layers': 6,
#                'dropout': 0.2,
#                'lr': 3e-4,
#                'batch_size': 64,
#                'num_epochs': 10}

model_paras = {'embedding_size': 32,
               'head_size': 8,
               'vocab_size': vocab_size,
               'block_size': 8,
               'num_heads': 4,
               'num_layers': 3,
               'dropout': 0,
               'lr': 1e-3,
               'batch_size': 64,
               'num_epochs': 3}


# data = token_encoder(text)
# # convert to tensor
# data = torch.tensor(data, dtype=torch.long)
# train_data = data[:int(len(data)*0.9)]
# train_set = TextDataset(train_data, model_paras['block_size'])
# val_data = data[int(len(data)*0.9):]
# val_set = TextDataset(val_data, model_paras['block_size'])

data = token_encoder(text)
# convert to tensor
data = torch.tensor(data, dtype=torch.long)



# print(model)


if __name__ == '__main__':
    
    # total_cpu_cores = mp.cpu_count()
    
    
    model = BigramModel(embedding_size=model_paras['embedding_size'], 
                        head_size=model_paras['head_size'], 
                        vocab_size=vocab_size, 
                        block_size=model_paras['block_size'], 
                        num_heads=model_paras['num_heads'], 
                        num_layers=model_paras['num_layers'], 
                        dropout=model_paras['dropout'])
    
    
    train_data = data[:int(len(data)*0.9)]
    train_set = TextDataset(train_data, model_paras['block_size'])

    
    # Initialize the multiprocessing environment
    mp.set_start_method('spawn')

    # Get the number of CPU cores available on the machine
    # num_processes = mp.cpu_count()
    num_processes = 4
    mp.spawn(train, args=(num_processes, model, train_set, model_paras), nprocs=num_processes, join=True)
    

    # # Start the training processes
    # processes = []
    # for rank in range(num_processes):
    #     p = mp.Process(target=train, args=(rank, num_processes, model_paras))
    #     p.start()
    #     processes.append(p)

    # # Wait for all processes to finish
    # for p in processes:
    #     p.join()