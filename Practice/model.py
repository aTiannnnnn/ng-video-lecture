import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


# dataset
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
    


# Model

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


