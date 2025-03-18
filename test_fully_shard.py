import torch
import torch.nn as nn

from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

# Define a simple model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# define another model with shared parameters beween fc1 and fc2
class SharedModel(nn.Module):
    def __init__(self):
        super(SharedModel, self).__init__()
        self.fc1 = nn.Linear(4, 2, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1, bias=False)
        self.fc2.weight = self.fc1.weight
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# set up distributed environment with cpu process group
torch.distributed.init_process_group("gloo")
model = Model()
fully_shard(model)

# Create some synthetic training data
batch_size = 8
num_batches = 5
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in range(num_batches):
        # Generate synthetic data
        inputs = torch.randn(batch_size, 4).to(device)
        targets = torch.randn(batch_size, 1).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        epoch_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    # Print statistics
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}")

print("Training complete!")

# Test the trained model
with torch.no_grad():
    test_input = torch.randn(1, 4).to(device)
    prediction = model(test_input)
    print(f"Test input: {test_input}")
    print(f"Model prediction: {prediction}")
