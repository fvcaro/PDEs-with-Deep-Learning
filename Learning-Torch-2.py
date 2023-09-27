import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np

# 1D heat equation
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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

pinn = PINN(2, [64, 128, 128, 128, 64], 1, act=nn.Sigmoid(), device=device)
print(pinn)

learning_rate = 1e-2
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3000, gamma=1e-1)

epochs = int(9e3)
convergence_data = torch.empty((epochs), device=device)

T = 2
boundary_condition_weight = 100.
initial_condition_weight = 100.

def random_domain_points(T, n=2048, device=torch.device("cpu")):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return x, t

def random_BC_points(T, n=128, device=torch.device("cpu")):
    x = torch.randint(2, (n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return x, t

def random_IC_points(n=32, device=torch.device("cpu")):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    t = T * torch.zeros(n, 1, device=device, requires_grad=True)
    return x, t

loss_list = []
loss_domain_list = []
loss_bc_list = []
loss_ic_list = []

for epoch in range(int(epochs)):
    optimizer.zero_grad()

    x, t = random_domain_points(T, device=device)
    u = pinn(x, t)

    u_t = torch.autograd.grad(outputs=u, 
                              inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_x = torch.autograd.grad(outputs=u, 
                              inputs=x,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_xx = torch.autograd.grad(outputs=u_x, 
                               inputs=x,
                               create_graph=True,
                               grad_outputs=torch.ones_like(u_x)
                               )[0]
    residual = u_t - u_xx
    loss_dom = torch.mean(torch.pow(residual, 2))
    
    x_bc, t_bc = random_BC_points(T, device=device)
    u_bc = pinn(x_bc, t_bc)
    loss_bc = torch.mean(torch.pow(u_bc - 0., 2))
    
    x_ic, t_ic = random_IC_points(device=device)
    u_ic = pinn(x_ic, t_ic)
    loss_ic = torch.mean(torch.pow(u_ic - torch.sin(torch.pi * x_ic), 2))
    
    loss = loss_dom + boundary_condition_weight * loss_bc + initial_condition_weight * loss_ic
    
    #loss_list.append(loss.detach().numpy())
    loss_list.append(loss.detach().cpu().numpy())
    loss_domain_list.append(loss_dom.detach().cpu().numpy())
    loss_bc_list.append(loss_bc.detach().cpu().numpy())
    loss_ic_list.append(loss_ic.detach().cpu().numpy()) 
    
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

x = torch.linspace(0, 1, 126).view(-1, 1).to(device)

for idx, t_i in enumerate(np.linspace(0, 2, 11)):
    t = t_i * torch.ones_like(x)
    nn_sol = pinn(x, t).detach().cpu().numpy()
    
    plt.figure()
    plt.plot(x.cpu().numpy(), nn_sol, label='nn')
    exact_sol = torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)
    plt.plot(x.cpu().numpy(), exact_sol.cpu().numpy(), label='exact sol')
    plt.title(r'$t_i$:' + str(t_i))
    plt.legend()

    plt.savefig(f'exact_plot_{idx}.png')

print('Plots saved successfully.')