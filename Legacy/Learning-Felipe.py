import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.layer2 = nn.Linear(2, 2)
        self.layer3 = nn.Linear(2, 3)
        self.layer4 = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Create an instance of the Model
model = Model()
model.to(device)
# Print the model architecture
print(model)
print(model.state_dict())
# Define the loss function and optimizer
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.05)
optimizer = optim.SGD(model.parameters(), lr=0.05)
# Some input data
input_data = [[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]
x = torch.tensor(input_data, requires_grad=True).view(2, 10)
print('input_data: ', x)
print('tensor    : ', x.shape)

target_data = [[1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]]
y = torch.tensor(target_data, requires_grad=True).view(2, 10)  # Adjusted target_data to have size (2, 2)
print('target_data:', y)
print('tensor     :', y.shape)
x = torch.transpose(x,1,0)
y = torch.transpose(y,1,0)
print('transpose: ', y)
# Define the number of epochs
num_epochs = int(1e6)
# Training loop
for epoch in range(num_epochs):
    # Move the tensors to the appropriate device
    x = x.to(device)
    y = y.to(device)
    k = torch.randint(0,10,(1,))  # choose a training point at random
    # print('k.item: ', k.item())
    x_aux = x[k.item()]
    # Forward pass
    outputs = model(x_aux)
    # Compute the loss
    loss = criterion(outputs, y[k.item()])
    # Zero the gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # Update the model's parameters
    optimizer.step()
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
# Print the final model's parameters
# print(model.state_dict())
for i in range(5):
    plt.plot(model(x[i]).detach().numpy()[0],model(x[i]).detach().numpy()[1],'ro')
for i in range(5,10):
    plt.plot(model(x[i]).detach().numpy()[0],model(x[i]).detach().numpy()[1],'gx')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.show()