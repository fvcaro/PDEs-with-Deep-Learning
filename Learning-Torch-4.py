import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np

# 1D Allen-Cahn equation PINN
class PINN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act=nn.ReLU(), device=torch.device("cpu")):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]).to(device))

        self.activation = act.to(device)
        self.device = device

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        output = self.layers[-1](inputs)
        return output

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

pinn = PINN(2, [128, 128, 128, 128], 1, act=nn.Tanh(), device=device)
print(pinn)

learning_rate = 1e-2
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=2500, gamma=1e-2)

epochs = int(1e4)
convergence_data = torch.empty((epochs), device=device)

T = 2
boundary_condition_weight = 100.
initial_condition_weight = 50.

def random_domain_points(T, n=16384, device=torch.device("cpu")):
    x = torch.rand(n, 1, device=device, requires_grad=True) * 2 - 1  # x in [-1, 1]
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return x, t

def random_BC_points(T, n=512, device=torch.device("cpu")):
    x = torch.randint(2, (n, 1), dtype=torch.float, device=device, requires_grad=True) * 2 - 1
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return x, t

def random_IC_points(n=1024, device=torch.device("cpu")):
    x = torch.rand(n, 1, device=device, requires_grad=True) * 2 - 1  # x in [-1, 1]
    t = T * torch.zeros(n, 1, device=device, requires_grad=True)
    return x, t

loss_list = []
loss_domain_list = []
loss_bc_list = []
loss_ic_list = []

for epoch in range(int(epochs)):
    optimizer.zero_grad()

    # Residual Loss
    x, t = random_domain_points(T, device=device)
    u = pinn(x, t)
    u_t = torch.autograd.grad(outputs=u, inputs=t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_x = torch.autograd.grad(outputs=u, inputs=x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
    residual = u_t - 0.0001 * u_xx + 5 * torch.pow(u, 3) - 5 * u
    loss_dom = torch.mean(torch.pow(residual, 2))

    # Boundary Condition Loss
    x_bc, t_bc = random_BC_points(T, device=device)
    u_left = pinn(-torch.ones_like(t_bc), t_bc)
    u_right = pinn(torch.ones_like(t_bc), t_bc)
    x_left = -torch.ones_like(t_bc, requires_grad=True)
    x_right = torch.ones_like(t_bc, requires_grad=True)

    u_left = pinn(x_left, t_bc)
    u_right = pinn(x_right, t_bc)

    u_x_left = torch.autograd.grad(outputs=u_left, inputs=x_left, create_graph=True, grad_outputs=torch.ones_like(u_left))[0]
    u_x_right = torch.autograd.grad(outputs=u_right, inputs=x_right, create_graph=True, grad_outputs=torch.ones_like(u_right))[0]

    loss_bc = torch.mean(torch.pow(u_left - u_right, 2)) + torch.mean(torch.pow(u_x_left - u_x_right, 2))

    # Initial Condition Loss
    x_ic, t_ic = random_IC_points(device=device)
    u_ic = pinn(x_ic, t_ic)
    loss_ic = torch.mean(torch.pow(u_ic - x_ic**2 * torch.cos(torch.pi * x_ic), 2))

    loss = loss_dom + boundary_condition_weight * loss_bc + initial_condition_weight * loss_ic

    loss_list.append(loss.detach().numpy())
    loss_domain_list.append(loss_dom.detach().numpy())
    loss_bc_list.append(loss_bc.detach().numpy())
    loss_ic_list.append(loss_ic.detach().numpy())

    loss.backward()
    optimizer.step()
    scheduler.step()

    convergence_data[epoch] = loss.item()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")

# Plotting individual losses
plt.figure(figsize=(10, 6))
plt.semilogy(loss_list, label='Total Loss')
plt.semilogy(loss_domain_list, label='Domain Loss')
plt.semilogy(loss_bc_list, label='Boundary Condition Loss')
plt.semilogy(loss_ic_list, label='Initial Condition Loss')
plt.legend()
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True, which="both", ls="--")
plt.savefig('individual_losses_plot.png')

x = torch.linspace(-1, 1, 512).view(-1, 1)

for idx, t_i in enumerate(np.linspace(0.1, 2.1, 11)):  # Adjusted the start and end values of linspace
    t = t_i * torch.ones_like(x)
    nn_sol = pinn(x, t).detach().numpy()
    
    plt.figure()
    plt.plot(x, nn_sol, label='nn')
    plt.title(r'$t_i$:' + str(t_i))
    plt.legend()

    plt.savefig(f'nn_plot_{idx}.png')

# Define a meshgrid for t and x values
t_vals = np.linspace(0, 1, 100)
x_vals = np.linspace(-1, 1, 100)
t_grid, x_grid = np.meshgrid(t_vals, x_vals)

# Convert these to PyTorch tensors
t_tensor = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float, device=device)
x_tensor = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float, device=device)

# Get the model's predictions
u_preds = pinn(x_tensor, t_tensor)
u_preds = u_preds.detach().cpu().numpy().reshape(t_grid.shape)

# Plot the results
plt.figure(figsize=(10, 6))
plt.imshow(u_preds, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='RdBu')  # Changed cmap to RdBu
plt.colorbar(label='$u(t,x)$')
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('$u(t,x)$')
plt.savefig('u_tx_plot.png', dpi=300)  # This line saves the plot to an image
plt.show()

print('Plots saved successfully.')