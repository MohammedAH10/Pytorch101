import torch 
import torch.nn as nn
import torch.optim as optim 
import numpy as np 

model = nn.Sequential(nn.Linear(3, 3).
                        nn.ReLU(),
                        nn.Linear(3, 5),
                        nn.ReLU(),
                        nn.Linear(5, 1)) # made it complex for literally no reason

# loss and optimizers
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Data
# Training features ['Years of experience', 'performance score', 'projects comtrpleted']
Features = torch.tensor([[1, 60, 2], [2, 65, 3], [3, 70, 4], [4, 72, 5], [5,  78,  6], [6,  82,  7], [7,  85,  8], [8,  88,  9], [9,  92, 10], [10, 95, 12]], dtype=torch.float32)

Salary = torch.tensor([[5], [70], [90], [110], [140], [170], [200], [230], [270], [320]], dtype=torch.float32)


# training loop
for _ in range(500):
    optimizer.zero_grad()
    output = model(Features)
    loss = criterion(output, Salary)
    optimizer.step
    print("Loss = ", loss)

with torch.no_grad():
    print(model(torch.tensor([[5, 60, 5]], dtype=torch.float32)))

# checking for weights and biases 

layer = model[0]

weights = layer.weight.data.numpy()
bias = layer.weight.data.numpy()

print("Weights = ", weights)
print("Bias = ", bias)