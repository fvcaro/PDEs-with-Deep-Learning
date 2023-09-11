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
        inputs = torch.cat([x, t], axis=1)  # Concatenate x and t
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

learning_rate = 3e-3
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3000, gamma=1e-5)  # Learning rate scheduler

epochs = int(7e3)
convergence_data = torch.empty((epochs), device=device)

T = 2
gamma1 = 100.
gamma2 = 100.

def random_domain_points(T, n=2048):
    x = torch.rand(n,1,requires_grad=True)
    t = T*torch.rand(n,1,requires_grad=True)
    return x, t

def random_BC_points(T, n=128):
    x = torch.randint(2, (n,1), dtype=torch.float32, requires_grad=True)
    t = T*torch.rand(n,1,requires_grad=True)
    return x, t

def random_IC_points(n=32):
    x = torch.rand(n,1,requires_grad=True)
    t = T*torch.zeros(n,1,requires_grad=True)
    return x, t

loss_list = []
for epoch in range(int(epochs)):
    optimizer.zero_grad() # to make the gradients zero
    #
    x, t = random_domain_points(T)
    u = pinn(x, t)
    # Derivatives
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
    loss_dom = torch.mean(torch.pow(residual,2))
    # BC
    x_bc, t_bc = random_BC_points(T)
    u_bc = pinn(x_bc, t_bc)
    loss_bc = torch.mean(torch.pow(u_bc - 0.,2))
    # IC
    x_ic, t_ic = random_IC_points()
    u_ic = pinn(x_ic, t_ic)
    loss_ic = torch.mean(torch.pow(u_ic - torch.sin(torch.pi*x_ic),2))
    # LOSS
    loss = loss_dom + gamma1*loss_bc + gamma2*loss_ic
    loss_list.append(loss.detach().numpy())
    loss.backward() # This is for computing gradients using backward propagation
    optimizer.step() 
    scheduler.step()  # Update learning rate

    # Store loss into convergence_data and print loss and learning rate every 100 epochs
    convergence_data[epoch] = loss.item()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")

plt.semilogy(loss_list)
plt.savefig('loss_plot.png')

x = torch.linspace(0,1,126).view(-1,1)

for idx, t_i in enumerate(np.linspace(0,2,11)):
    t = t_i * torch.ones_like(x)
    nn_sol = pinn(x,t).detach().numpy()
    
    plt.figure()
    plt.plot(x, nn_sol, label='nn')
    exact_sol = torch.sin(torch.pi*x) * torch.exp(-torch.pi**2*t)
    plt.plot(x, exact_sol, label='exact sol')
    plt.title(r'$t_i$:' + str(t_i))
    plt.legend()

    plt.savefig(f'exact_plot_{idx}.png')  # save with a unique name

print('Plots saved successfully.')