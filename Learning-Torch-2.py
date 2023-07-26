import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

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

def random_domain_points(n):
    x = torch.rand(n, 1, requires_grad=True)
    return x

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

pinn = PINN(1, [10, 50, 50, 50, 10], 1, act=torch.nn.Sigmoid(), device=device)
print(pinn)

learning_rate = 0.01
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)  # Learning rate scheduler

epochs = int(2e3)
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
        u_x = torch.autograd.grad(outputs=u, inputs=x, create_graph=True,
                                  grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(outputs=u_x, inputs=x, create_graph=True,
                                   grad_outputs=torch.ones_like(u_x))[0]

        # Compute the residual and loss in the domain
        residual = -u_xx - 4.*(torch.pi**2)*torch.sin(2.*torch.pi*x)
        loss_dom = torch.mean(torch.pow(residual, 2))

        # Compute the loss at the boundary
        x_bc = torch.tensor([[0.], [1.]], requires_grad=True).to(device)
        u_bc = pinn(x_bc)
        loss_bc = torch.mean(torch.pow(u_bc - 0., 2))

        # Total loss
        loss = loss_dom + gamma1 * loss_bc
        loss_list.append(loss.item())

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        convergence_data[epoch] = loss.item()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")

except KeyboardInterrupt:
    pass

plt.semilogy(loss_list)
plt.savefig('loss_plot.png')

x = torch.linspace(0, 1, 126).view(-1, 1).to(device)
nn_sol = pinn(x).detach().cpu().numpy()

plt.figure()
plt.plot(x.cpu().numpy(), nn_sol, label='nn')
exact_sol = torch.sin(2.*torch.pi*x).cpu().numpy()
plt.plot(x.cpu().numpy(), exact_sol, label='exact sol')
plt.ylim(-1.1, 1.1)
plt.legend()
plt.savefig('exact_plot.png')

print('Plots saved successfully.')
