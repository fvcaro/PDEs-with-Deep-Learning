import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os

# 1D wave eq. adaptive scenario
class PINN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act=nn.ReLU(), device=torch.device("cpu")):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            # Applying Xavier uniform initialization to the linear layers' weights
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)

        self.activation = act
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

torch.manual_seed(42)
pinn = PINN(2, [10, 50, 50, 50, 10], 1, act=nn.Tanh())
pinn = pinn.to(device)
print(pinn)

learning_rate = 1e-2
optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3000, gamma=1e-1)

# LBFGS as optimizer:
# optimizer = optim.LBFGS(pinn.parameters(), lr=learning_rate, max_iter=20, history_size=100) 
# The max_iter parameter specifies the maximum number of iterations the algorithm will take in a single batch. You might need to tune this.
# The history_size parameter specifies how many of the previous gradient evaluations should be stored. Older gradients are discarded.
# Lastly, when using the LBFGS optimizer, the way you perform the optimization in your training loop will be slightly different. Instead of the usual optimizer.zero_grad(), loss.backward(), and optimizer.step(), 
# you will typically define a closure function that computes the loss and its gradients and pass it to optimizer.step()

# def closure():
#     optimizer.zero_grad()
#     # compute your losses here
#     loss = ... 
#     loss.backward()
#     return loss

# optimizer.step(closure)

# epochs_list = [6000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,20000]
epochs_list = [2000,2000]

# convergence_data = torch.empty((epochs), device=device)

boundary_condition_weight = 100. # gamma1
initial_condition_weight = 100.  # gamma2

dom_points = 2048
bc_points  = 1128
ic_points  = 64

def random_domain_points(L, T, n=8192, device=torch.device("cpu")):
    x = (2*L)*torch.rand(n, 1, device=device, requires_grad=True) - L  
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return x, t

def random_BC_points(L, T, n=512, device=torch.device("cpu")):
    x = L * torch.randint(2, (n, 1), dtype=torch.float32, device=device, requires_grad=True) - L  
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return x, t

def random_IC_points(L, n=128, device=torch.device("cpu")):
    x = 2*L*torch.rand(n, 1, device=device, requires_grad=True) - L  
    t = torch.zeros(n, 1, device=device, requires_grad=True)
    return x, t

L     = 40
T     = 30

x_domain, t_domain = random_domain_points(L, T, device=device)
x_bc, t_bc = random_BC_points(L, T, device=device)
x_ic, t_ic = random_IC_points(L, device=device)

x0    = 0
A     = 1.
c     = 1.

def exact_sol_f(x,t,x0=x0,A=A,c=c):
    aux = torch.exp(-A*((x - x0) - c*t)**2)/2 + torch.exp(-A*((x - x0) + c*t)**2)/2
    return aux

# Always make sure to .cpu() tensors before converting them to numpy for plotting to avoid issues with CUDA tensors
x = torch.linspace(-L, L, 256, device=device).view(-1, 1)

# Define a directory to save the figures
save_dir = 'Figs'
os.makedirs(save_dir, exist_ok=True)

plt.figure()
exact_sol = exact_sol_f(x, 0.)
plt.plot(x.cpu().numpy(), exact_sol.cpu().numpy(), label='exact sol', color='tab:orange')
plt.legend()
save_filename = os.path.join(save_dir, 'wave_Exact_sol.png')
plt.savefig(save_filename, dpi=300)
plt.show()

plt.figure()
plt.plot(x_domain.cpu().detach().numpy(), t_domain.cpu().detach().numpy(), 'o', ms=1)
plt.plot(x_ic.cpu().detach().numpy(), t_ic.cpu().detach().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'points.png')
plt.savefig(save_filename, dpi=300)
plt.show()

def loss_1(x,t):
    u = pinn(x,t)
    # Derivatives
    u_t  = torch.autograd.grad(outputs=u, 
                              inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_tt = torch.autograd.grad(outputs=u_t, 
                              inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_t)
                              )[0]
    u_x  = torch.autograd.grad(outputs=u, 
                              inputs=x,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_xx = torch.autograd.grad(outputs=u_x, 
                               inputs=x,
                               create_graph=True,
                               grad_outputs=torch.ones_like(u_x)
                               )[0]
    residual = u_tt - (c**2)*u_xx
    loss_dom = torch.mean(torch.pow(residual,2))
    return loss_dom

def loss_2(x_bc, t_bc):
   x_bc = x_bc.to(device)  # Move x_bc to the same device as pinn
   t_bc = t_bc.to(device)  # Move t_bc to the same device as pinn
   u_bc    = pinn(x_bc, t_bc)
   u_bc_t  = torch.autograd.grad(outputs=u_bc, 
                             inputs=t_bc,
                             create_graph=True,
                             grad_outputs=torch.ones_like(u_bc)
                             )[0]
   u_bc_x  = torch.autograd.grad(outputs=u_bc, 
                             inputs=x_bc,
                             create_graph=True,
                             grad_outputs=torch.ones_like(u_bc)
                             )[0]
   # loss_bc = torch.mean(torch.pow(x_bc * u_bc_t - u_bc + x_bc * u_bc_x, 2))
   loss_bc = torch.mean(torch.pow(u_bc - 0., 2))
   return loss_bc

def loss_3(x_ic, t_ic):
    x_ic = x_ic.to(device)  # Move x_ic to the same device as pinn
    t_ic = t_ic.to(device)  # Move t_ic to the same device as pinn
    u_ic = pinn(x_ic, t_ic)

    u_ic_t = torch.autograd.grad(outputs=u_ic, 
                                 inputs=t_ic,
                                 create_graph=True,
                                 grad_outputs=torch.ones_like(u_ic)
                                 )[0]

    loss_ic = torch.mean(torch.pow(u_ic - torch.exp(-A * torch.pow(x_ic, 2)), 2))
    loss_ic += torch.mean(torch.pow(u_ic_t - 0., 2))
    return loss_ic

loss_list = []
t0 = time()
# Initial training set
t = t_domain  # Initialize t with the domain points

for epochs in epochs_list:
    for epoch in range(int(epochs)):
        # Track epochs
        #if (epoch%(epochs/10)==0):
        #    print('epoch:',epoch)
        optimizer.zero_grad() # to make the gradients zero
        # RESIDUAL ################################################################   
        loss_dom = loss_1(x_domain, t_domain)
        # BC ######################################################################
        loss_bc  =  loss_2(x_bc,t_bc)
        # IC ######################################################################
        loss_ic  = loss_3(x_ic,t_ic)
        # LOSS ####################################################################
        loss = loss_dom + boundary_condition_weight*loss_bc + initial_condition_weight*loss_ic
        loss_list.append(loss.detach().cpu().numpy())
        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # 
        scheduler.step()  # Update learning rate
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")
    # Adative sample step    
    x_,t_        = random_domain_points(L, T, n=10*dom_points, device=device)
    loss_dom_aux = torch.zeros_like(x_)
    #
    x_bc_, t_bc_ = random_BC_points(L, T, n=10*bc_points, device=device)
    loss_bc_aux  = torch.zeros_like(x_bc_)
    #
    x_ic_, t_ic_ = random_IC_points(L, n=10*ic_points, device=device)
    loss_ic_aux = torch.zeros_like(x_ic_)
    #
    for idx in range(x_.shape[0]):
        loss_dom_aux[idx] = loss_1(x_[idx:idx+1],t_[idx:idx+1])
    for idx in range(x_bc_.shape[0]):
        loss_bc_aux[idx] = loss_2(x_bc_[idx:idx+1],t_bc_[idx:idx+1])
    for idx in range(x_ic_.shape[0]):
        loss_ic_aux[idx] = loss_3(x_ic_[idx:idx+1],t_ic_[idx:idx+1])    
    # 
    # [16] proposes to update the training dataset by selecting 
    # points from a uniformly distributed set with larger residual values
    idx_dom = torch.where(loss_dom_aux >= loss_dom_aux.sort(0)[0][-500])
    idx_bc  = torch.where(loss_bc_aux >= loss_bc_aux.sort(0)[0][-10])
    idx_ic  = torch.where(loss_ic_aux >= loss_ic_aux.sort(0)[0][-50])
    #
    x_aux = x_[idx_dom].view(-1,1)
    #print(x_[idx_dom])
    t_aux = t_[idx_dom].view(-1,1)
    x = torch.cat((x,x_aux),0)
    t = torch.cat((t,t_aux),0)
    #
    x_bc_aux = x_bc_[idx_bc].view(-1, 1)
    t_bc_aux = t_bc_[idx_bc].view(-1, 1)

    # Move x_bc_aux and t_bc_aux to the same device as x_bc
    x_bc_aux = x_bc_aux.to(x_bc.device)
    t_bc_aux = t_bc_aux.to(x_bc.device)

    x_bc = torch.cat((x_bc, x_bc_aux), 0)
    t_bc = torch.cat((t_bc, t_bc_aux), 0)
    #
    x_ic_aux = x_ic_[idx_ic].view(-1, 1)
    t_ic_aux = t_ic_[idx_ic].view(-1, 1)

    # Move x_ic_aux and t_ic_aux to the same device as x_ic
    x_ic_aux = x_ic_aux.to(x_ic.device)
    t_ic_aux = t_ic_aux.to(x_ic.device)

    x_ic = torch.cat((x_ic, x_ic_aux), 0)
    t_ic = torch.cat((t_ic, t_ic_aux), 0)
    #
    # keep editing from here (code the criteria to select the point with biggest loss value)
    #
print('computing time',(time() - t0)/60,'[min]') 

plt.figure()
plt.plot(x.cpu().detach().numpy(), 'o', ms=1)
plt.plot(t.cpu().detach().numpy(), 'o', ms=1)
plt.plot(x_bc.cpu().detach().numpy(), 'o', ms=1)
plt.plot(t_bc.cpu().detach().numpy(), 'o', ms=1)
plt.plot(x_ic.cpu().detach().numpy(), 'o', ms=1)
plt.plot(t_ic.cpu().detach().numpy(), 'o', ms=1)
plt.show()

plt.figure()
plt.semilogy(loss_list)
save_filename = os.path.join(save_dir, 'wave_loss_list.png')
plt.savefig(save_filename, dpi=300)

for t_i in np.linspace(0, T, 11):
    t = t_i * torch.ones_like(x)
    nn_sol = pinn(x, t).detach().cpu().numpy()

    plt.figure()
    plt.plot(x.cpu().detach().numpy(), nn_sol, label='nn', linestyle='', marker='o', linewidth=2)
    exact_sol = exact_sol_f(x, t_i)
    plt.plot(x.cpu().detach().numpy(), exact_sol.cpu().detach().numpy(), color='tab:orange', linestyle='', marker='o', label='exact sol')
    plt.title(r'$t_i$: ' + str(t_i))
    plt.xlim(-L, L)
    plt.legend()

    # Save the figure with a unique name based on the time step
    save_filename = os.path.join(save_dir, f'figure_t_{t_i}.png')
    plt.savefig(save_filename, dpi=300)  # You can adjust the dpi (dots per inch) for higher resolution
    plt.close()  # Close the figure to release resources