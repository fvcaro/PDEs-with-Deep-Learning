import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
#
# Define a directory to save the figures
save_dir = 'Figs'
os.makedirs(save_dir, exist_ok=True)
#
## Parameters
LENGTH = 40. # Domain size in x axis. 
TOTAL_TIME = 30. # Domain size in t axis. 
WAVE_SPEED = 1.
INITIAL_SHAPE = 1.
INITIAL_POSITION = 0.

LAYERS = 10
NEURONS_PER_LAYER = 120
LEARNING_RATE = 0.0015

#EPOCHS = 1500
epochs_list = [1500]
WEIGHT_RESIDUAL = 100. # Weight of residual part of loss function
WEIGHT_INITIAL = 100. # Weight of initial part of loss function

DOM_POINTS = 1024
IC_POINTS = 32

def u(x: torch.Tensor, t: torch.Tensor, WAVE_SPEED, INITIAL_SHAPE, INITIAL_POSITION) -> torch.Tensor:
    """Compute the exact solution"""
    return torch.exp(-INITIAL_SHAPE*((x - INITIAL_POSITION) - WAVE_SPEED*t)**2)/2 + torch.exp(-INITIAL_SHAPE*((x - INITIAL_POSITION) + WAVE_SPEED*t)**2)/2

x_exact = torch.linspace(-LENGTH,LENGTH,256).view(-1,1)
#
plt.figure()
exact_sol = u(x_exact, 0., WAVE_SPEED, INITIAL_SHAPE, INITIAL_POSITION)
plt.plot(x_exact, exact_sol, label='exact_sol', color='tab:orange')
plt.legend()
save_filename = os.path.join(save_dir, 'exact_sol.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

class PINN(nn.Module):
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):

        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device

def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, t)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    for _ in range(order):
        grads = torch.autograd.grad(
            output,
            input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]
        output = grads
    return grads

def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)

def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)

def residual_loss(pinn: PINN, x_domain, t_domain, WAVE_SPEED):
    loss = dfdt(pinn, x_domain, t_domain, order=2) - WAVE_SPEED**2 * dfdx(pinn, x_domain, t_domain, order=2)
    return loss.pow(2).mean()

def initial_loss(pinn: PINN, x_domain, t_domain, INITIAL_SHAPE):
    # Compute the exact initial condition at the given x points
    u_exact_init = torch.exp(-INITIAL_SHAPE * x_domain.pow(2))
    # Evaluate the PINN model at the initial time points
    u_pred = f(pinn, x_domain, t_domain)
    # Compute the time derivative of the PINN output using dfdt
    u_pred_t = dfdt(pinn, x_domain, t_domain)
    # Calculate the difference between the predicted values and the exact values for u and its t derivative
    loss_u = u_pred - u_exact_init
    loss_ut = u_pred_t
    # Compute the total loss by combining both losses
    total_loss = loss_u.pow(2).mean() + loss_ut.pow(2).mean()
    return total_loss

def random_domain_points(LENGTH,TOTAL_TIME,n=8192):
    x = (2*LENGTH)*(torch.rand(n,1,requires_grad=True) - 0.5).to(device)
    t = TOTAL_TIME*torch.rand(n,1,requires_grad=True).to(device)
    return x, t

def random_IC_points(LENGTH,n=128):
    x = (2*LENGTH)*(torch.rand(n,1,requires_grad=True) - 0.5).to(device)
    t = TOTAL_TIME*torch.rand(n,1,requires_grad=True).to(device)
    return x, t

x, t        = random_domain_points(LENGTH,TOTAL_TIME, n=DOM_POINTS)
x_ic, t_ic  = random_IC_points(LENGTH, n=IC_POINTS)

plt.figure()
plt.plot(x.detach().cpu().numpy(), t.detach().cpu().numpy(), 'o', ms=1)
plt.plot(x_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)  # Added .cpu() here
save_filename = os.path.join(save_dir, 'points.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

## Running code
pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)
print(pinn)

optimizer = torch.optim.Adam(pinn.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=2000, gamma=1., verbose=False)  # Learning rate scheduler

loss_list = []
t0 = time()
# Initial training set

for epochs in epochs_list:
    for epoch in range(int(epochs)):
        optimizer.zero_grad() # to make the gradients zero
        loss_dom = residual_loss(pinn, x, t, WAVE_SPEED)
        loss_ic  = initial_loss(pinn, x_ic, t_ic, INITIAL_SHAPE)
        loss = WEIGHT_RESIDUAL*loss_dom + WEIGHT_INITIAL*loss_ic
        loss_list.append(loss.detach().cpu().numpy())
        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # 
        scheduler.step()  # Update learning rate
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")
print('computing time',(time() - t0)/60,'[min]') 