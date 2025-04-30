import torch

import torch.nn as nn

class TwoLinearModule(nn.Module):
    def __init__(self, in_features=10, hidden_features=20, out_features=5):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Instantiate the model
model = TwoLinearModule()
print("Original model:")
print(model)
original_modules = model._modules
print("\nOriginal modules:", original_modules)

# Delete the first linear module
deleted_linear = model.linear1
delattr(model, 'linear1')
print("\nModel after deletion:")
print(original_modules)
print(model)

# Add the deleted linear module back
model.linear1 = deleted_linear
print(original_modules)
model._modules = original_modules
print("\nModel after restoration:")
print(model._modules)