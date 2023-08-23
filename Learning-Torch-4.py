import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
import numpy as np

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PINN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act=Swish(), device=torch.device("cpu")):
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

def random_domain_points(n):
    x = torch.rand(n, 2, requires_grad=True)
    return x

def f(x, y):
    return 2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

def exact_solution(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

pinn = PINN(2, [64, 128, 128, 128, 64], 1, act=Swish(), device=device)
print(pinn)

learning_rate = 0.001
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

epochs = int(5e3)
convergence_data = torch.empty((epochs), device=device)

gamma1 = 100.
loss_list = []

try:
    for epoch in range(epochs):
        optimizer.zero_grad()  # Set gradients to zero

        # Random domain points
        x = random_domain_points(2048).to(device)
        u = pinn(x)

        # Compute derivatives
        x.requires_grad = True
        u = pinn(x)
        u_x, u_y = torch.autograd.grad(outputs=u, inputs=x, create_graph=True, grad_outputs=torch.ones_like(u), 
                                       retain_graph=True)[0].split(1, dim=1)

        u_xx = torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x), 
                                   retain_graph=True)[0]
        u_yy = torch.autograd.grad(outputs=u_y, inputs=x, grad_outputs=torch.ones_like(u_y), 
                                   retain_graph=True)[0]

        # Compute the residual and loss in the domain
        residual = -(u_xx + u_yy) - f(x[:, 0], x[:, 1]).view(-1, 1)
        loss_dom = torch.mean(torch.pow(residual, 2))

        # Compute the loss at the boundary (zero Dirichlet boundary conditions)
        x_bc = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], requires_grad=True).to(device)
        u_bc = pinn(x_bc)
        loss_bc = torch.mean(torch.pow(u_bc, 2))

        # Total loss
        loss = loss_dom + gamma1 * loss_bc
        loss_list.append(loss.item())

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update learning rate based on validation loss
        val_loss = loss.item()
        scheduler.step(val_loss)

        convergence_data[epoch] = val_loss

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} - Loss: {val_loss:.6f} - Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.8f}")

except KeyboardInterrupt:
    pass

plt.semilogy(loss_list)
plt.savefig('loss_plot.png')

# Plotting the 2D solution
n_points = 100
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