import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the distributed environment
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(29000)
dist.init_process_group('gloo', rank=0, world_size=1)
#dist.init_process_group(backend='gloo', init_method='env://')
rank = dist.get_rank()
print(f"rank={rank}")
#import sys
#sys.exit()
world_size = dist.get_world_size()

# Set the number of worker processes and CPU cores per process
num_workers = 4
num_cores = 2

# Set the number of batches per worker process
num_batches = 10

# Compute the batch size per core
batch_size = 32
batch_size_per_core = batch_size // num_cores

# Compute the total batch size per worker process
total_batch_size = batch_size_per_core * num_cores

# Define your model
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

# Wrap your model in DDP
model = DDP(model)

# Define your dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 10)
        self.targets = torch.randn(1000, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

dataset = MyDataset()

# Define your dataloader with multiple worker processes
dataloader = DataLoader(dataset, batch_size=total_batch_size, shuffle=True, num_workers=num_workers)

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
import time
# Train your model
for epoch in range(10):
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        if batch_idx % world_size != rank:
            continue
        # Split the batch across the CPU cores
        inputs = [inputs[i:i+batch_size_per_core, :] for i in range(0, batch_size, batch_size_per_core)]
        targets = [targets[i:i+batch_size_per_core, :] for i in range(0, batch_size, batch_size_per_core)]
        # Send inputs and targets to the current device
        inputs = [input.to(f'cpu:{rank}') for input in inputs]
        targets = [target.to(f'cpu:{rank}') for target in targets]

        # Compute the outputs
        outputs = [model(input) for input in inputs]

        # Compute the loss
        loss = sum([criterion(output, target) for output, target in zip(outputs, targets)])

        # Compute gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()        
    print(f'epoch {epoch} time: ', (end_time - start_time)*100, 'loss: ', loss.item())

    # Synchronize the model parameters across all processes
dist.barrier()
if rank == 0:
    dist.broadcast(model.state_dict(), src=0)
