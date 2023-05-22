"""
@author: Felipe V. Caro
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the neural network
model = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate some sample input data
input_data = torch.randn(100, 10)
target_data = torch.randn(100, 1)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Make predictions on new data
new_data = torch.randn(10, 10)
predictions = model(new_data)
print("Predictions:")
print(predictions)

