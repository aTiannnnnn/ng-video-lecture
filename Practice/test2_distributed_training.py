import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)
import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from model import *

def clear_up():
    """ Clear up the environment. """
    dist.destroy_process_group()

# clear_up()
def init_process(rank, world_size, backend='gloo'):
    """ Initialize the distributed environment. """
    # set a random port number
    port_number = 38021
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port_number)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def train(rank, world_size, model, train_set, val_set, model_paras, num_workers=0):
    
    init_process(rank, world_size)
    device = torch.device(f"cpu:{rank}")
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=model_paras['batch_size'], sampler=train_sampler, num_workers=num_workers)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_set, batch_size=model_paras['batch_size'], sampler=val_sampler, num_workers=num_workers)
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    eval_interval = model_paras['eval_interval']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_paras['lr'])
    
    model.to(device)
    model = DDP(model)
    
    model.train()
    for epoch in range(model_paras['num_epochs']):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        total_train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x,y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            step_idx = epoch * num_train_batches + batch_idx
            
            if batch_idx % eval_interval == 0:
                train_loss = total_train_loss / (batch_idx + 1)
                
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        _, val_loss = model(x,y)
                        total_val_loss += val_loss.item()
                val_loss = total_val_loss / num_val_batches
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Step: {step_idx}, Train Loss: {train_loss}, Val Loss: {val_loss}")
                model.train()
        
        end_time = time.time()
        duration = end_time - start_time
        batches_per_sec = num_train_batches / duration
        print(f"Epoch: {epoch}, Batches per second: {batches_per_sec}")
    
def train_without_val(rank, world_size, model, train_set, model_paras, num_workers=0):
    init_process(rank, world_size)
    device = torch.device(f"cpu:{rank}")
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=model_paras['batch_size'], sampler=train_sampler, num_workers=num_workers)
    num_train_batches = len(train_loader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_paras['lr'])
    
    model.to(device)
    model = DDP(model)
    
    model.train()
    for epoch in range(model_paras['num_epochs']):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        total_train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            start_time_batch = time.time()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x,y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            end_time_batch = time.time()
            print(f"{batch_idx}th Batch time: ", end_time_batch - start_time_batch)
        
        end_time = time.time()
        duration = end_time - start_time
        batches_per_sec = num_train_batches / duration
        print(f"Epoch: {epoch}, Batches per second: {batches_per_sec}")
        

def train_without_val(rank, world_size, model, train_set, model_paras, num_workers=0):
    init_process(rank, world_size)
    device = torch.device(f"cpu:{rank}")
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=model_paras['batch_size'], sampler=train_sampler, num_workers=num_workers)
    num_train_batches = len(train_loader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_paras['lr'])
    eval_interval = model_paras['eval_interval']
    
    model.to(device)
    # model = DDP(model)
    
    model.train()
    for epoch in range(model_paras['num_epochs']):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        total_train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            start_time_batch = time.time()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x,y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            end_time_batch = time.time()
            step_idx = epoch * num_train_batches + batch_idx

            if step_idx % eval_interval == 0:
                print(f"{batch_idx}th Batch time: ", end_time_batch - start_time_batch)
        
        end_time = time.time()
        duration = end_time - start_time
        batches_per_sec = num_train_batches / duration
        print(f"Epoch: {epoch}, Batches per second: {batches_per_sec}")
        


if __name__ == '__main__':
    # load data
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
    data = token_encoder(text)
    data = torch.tensor(data, dtype=torch.long)
    train_data = data[:int(0.9 * len(data))]
    val_data = data[int(0.9 * len(data)):]

    # set hyperparameters
    # 62 batches
    model_paras_large = {'embedding_size': 384,
                         'head_size': 64,
                         'vocab_size': vocab_size,
                         'block_size': 256,
                         'num_heads': 6,
                         'num_layers': 6,
                         'dropout': 0.2,
                         'lr': 3e-4,
                         'batch_size': 64,
                         'num_epochs': 10,
                         'eval_interval': 4,
                         'max_new_tokens': 500}

    # 3922 batches
    model_paras_small = {'embedding_size': 32,
                         'head_size': 8,
                         'vocab_size': vocab_size,
                         'block_size': 8,
                         'num_heads': 4,
                         'num_layers': 3,
                         'dropout': 0.0,
                         'lr': 1e-3,
                         'batch_size': 32,
                         'num_epochs': 4,
                         'eval_interval': 400,
                         'max_new_tokens': 500}
    
    model_paras = model_paras_small
    train_set = TextDataset(train_data, block_size=model_paras['block_size'])
    model = BigramModel(embedding_size=model_paras['embedding_size'], 
                        head_size=model_paras['head_size'], 
                        vocab_size=vocab_size, 
                        block_size=model_paras['block_size'], 
                        num_heads=model_paras['num_heads'], 
                        num_layers=model_paras['num_layers'], 
                        dropout=model_paras['dropout'])
    
    # Initialize the multiprocessing environment
    # mp.set_start_method('spawn')

    # Get the number of CPU cores available on the machine
    #num_processes = mp.cpu_count()
    num_workers  = 0
    num_processes = 1
    mp.spawn(train_without_val, args=(num_processes, model, train_set, model_paras, num_workers), nprocs=num_processes, join=True)

    # # Start the training processes
    
    # processes = []
    # model.share_memory()
    # for rank in range(num_processes):
    #     p = mp.Process(target=train_without_val, args=(rank, num_processes, model, train_set, model_paras))
    #     p.start()
    #     processes.append(p)

    # # # # Wait for all processes to finish
    # for p in processes:
    #     p.join()
    