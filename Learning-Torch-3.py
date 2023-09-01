import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np

class PINN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act=nn.ReLU(), device=torch.device("cpu")):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]).to(device))

        self.activation = act.to(device)
        self.device = device

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        output = self.layers[-1](x)
        return output
    
def random_dom_points(n):
    x = torch.rand(n, 2)
    x.requires_grad = True
    return x

def random_bc_points(n):
    n_ = int(n/4)
    x1 = torch.rand(n_,1)
    y1 = torch.zeros(n_,1)
    #
    x2 = torch.ones(n_,1)
    y2 = torch.rand(n_,1)
    #
    x3 = torch.rand(n_,1)
    y3 = torch.ones(n_,1)
    #
    x4 = torch.zeros(n_,1)
    y4 = torch.rand(n_,1)
    x = torch.cat((x1,x2,x3,x4),0)
    y = torch.cat((y1,y2,y3,y4),0)
    x.requires_grad = True
    y.requires_grad = True
    return x,y

def f(x, y):
    return 8 * (np.pi**2) * torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)

def exact_solution(x, y):
    return torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

pinn = PINN(2, [64, 128, 128, 128, 64], 1, act=torch.nn.ReLU(), device=device)
print(pinn)

learning_rate = 0.01
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning rate scheduler

epochs = int(2e3)
convergence_data = torch.empty((epochs), device=device)

gamma1 = 500.
loss_list = []

try:
    for epoch in range(epochs):
        optimizer.zero_grad()

        x = random_dom_points(4096)
        x = x.to(device)

        u = pinn(x)
        gradients = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = []
        u_yy = []

        u_xx_aux = torch.autograd.grad(gradients[:, 0], x, torch.ones_like(gradients[:, 0]), create_graph=True)[0]
        u_yy_aux = torch.autograd.grad(gradients[:, 1], x, torch.ones_like(gradients[:, 1]), create_graph=True)[0]
        u_xx=u_xx_aux[:,0]
        u_yy=u_yy_aux[:,1]

        residual = -(u_xx + u_yy).view(-1, 1) - f(x[:, 0], x[:, 1]).view(-1, 1)
        loss_dom = torch.mean(torch.pow(residual, 2))

        # Loss function (2nd component)
        x_bc,y_bc = random_bc_points(64)
        bc=torch.cat((x_bc,y_bc),1)
        u_bc = pinn(bc)
        loss_bc = torch.mean((u_bc - 0.)**2)

        loss = loss_dom + gamma1 * loss_bc
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        convergence_data[epoch] = loss.item()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")

except KeyboardInterrupt:
    pass

plt.semilogy(loss_list)
plt.savefig('loss_plot.png')

# Plotting the 2D solution
n_points = 1000
x, y = torch.meshgrid(torch.linspace(0, 1, n_points), torch.linspace(0, 1, n_points))
xy = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1).to(device)
nn_sol = pinn(xy).detach().cpu().numpy().reshape(n_points, n_points)
exact_sol = exact_solution(x, y).numpy()

plt.figure()
plt.contourf(x.numpy(), y.numpy(), nn_sol, levels=100, cmap='viridis')
plt.colorbar(label='PINN Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('PINN Solution')
plt.savefig('pinn_solution.png')

plt.figure()
plt.contourf(x.numpy(), y.numpy(), exact_sol, levels=100, cmap='viridis')
plt.colorbar(label='Exact Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exact Solution')
plt.savefig('exact_solution.png')

print('Plots saved successfully.')