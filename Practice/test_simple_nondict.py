import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the distributed environment

batch_size = 32
# Define your model
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
import time
# Train your model
for epoch in range(100):
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)
        # Compute gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    print(f'epoch {epoch} time: ', (end_time - start_time)*100, 'loss: ', loss.item())