# add self attention
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
# cd to the path of current file
os.chdir(current_path)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
max_new_tokens = 500

# add embedding layer and position embedding layer
embedding_size = 32
head_size = 32
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
data = torch.tensor(token_encoder(text), dtype=torch.long).to(device)
len_train = int(0.9 * len(data))
train_data = data[:len_train]
val_data = data[len_train:]

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() 
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        # x: (batch_size, block_size, embed_size)
        k = self.key(x) # (batch_size, block_size, head_size)
        q = self.query(x) # (batch_size, block_size, head_size)
        # compute attention scores
        wei = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5) # (batch_size, block_size, block_size)
        wei = wei.masked_fill(self.mask[:self.block_size, :self.block_size] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (batch_size, block_size, block_size)
        
        v = self.value(x) # (batch_size, block_size, head_size)
        out = torch.matmul(wei, v) # (batch_size, block_size, head_size)
        return out

# bigram model
class BigramModel(nn.Module):
    def __init__(self, embedding_size, head_size, vocab_size, block_size):
        super().__init__()
        self.token_embedding_tabel = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_tabel = nn.Embedding(block_size, embedding_size)
        self.sa_head = Head(head_size=head_size, embed_size=embedding_size, block_size=block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)
    
    def forward(self, inputs, targets=None):
        # inputs: (batch_size, block_size)
        # targets: (batch_size, block_size)
        batch_size, block_size = inputs.shape
        tok_emb = self.token_embedding_tabel(inputs) # (batch_size, block_size, embedding_size) / (B, T, C)
        pos_emb = self.position_embedding_tabel(torch.arange(block_size).to(device)) # (block_size, embedding_size) / (T, C) [same for all batch]
        x = tok_emb + pos_emb # (batch_size, block_size, embedding_size) / (B, T, C)
        x = self.sa_head(x) # (batch_size, block_size, head_size) / (B, T, C)
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


        
# test bigram model
model = BigramModel(embedding_size, head_size, vocab_size, block_size).to(device)
# train above model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
    
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,block_size), dtype=torch.long).to(device)
print(token_decoder(model.generate(context, max_new_tokens)[0].tolist()))
