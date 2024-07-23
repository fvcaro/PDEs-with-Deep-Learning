import torch
print('torch version:', torch.__version__)
import torch.nn as nn
import numpy as np
from time import time
import os
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)  # Set random seed for reproducibility
print('device:', device)

# PDE parameters
L  = 1
T  = 1

# Training parameters
GAMMA1 = 1.
GAMMA2 = 10.
#
TRAIN_DOM_POINTS = 262144 #131072
TRAIN_BC_POINTS  = 2048   #1024
TRAIN_IC_POINTS  = 2048
#
EPOCHS        = 500000
LEARNING_RATE = 0.001

# Define a directory to save the figures
save_dir = 'Figs_spherical_non_wave_eq'
os.makedirs(save_dir, exist_ok=True)

# NN class
class Model(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(),seed=42):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.seed = seed
        # Fix seed for reproducibility
        torch.manual_seed(seed)
        #
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Adding activation function for all but the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(self.activation)  
        # Initialize weights using Glorot initialization
        self.init_weights()  
    #
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Glorot initialization
                # nn.init.xavier_uniform_(layer.weight)  
                # Replace Glorot initialization with Kaiming initialization
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
                # Initialize bias to zeros
                nn.init.constant_(layer.bias, 0.0)  
    #
    def forward(self, x, y):
        #
        inputs = torch.cat([x, y], axis=1)
        #
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
def lossRes(r,t):
    u = model(r,t)
    # Derivatives
    u_t  = torch.autograd.grad(outputs=u, inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_tt = torch.autograd.grad(outputs=u_t, inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_t)
                              )[0]
    u_r  = torch.autograd.grad(outputs=u, inputs=r,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_rr = torch.autograd.grad(outputs=u_r, inputs=r,
                               create_graph=True,
                               grad_outputs=torch.ones_like(u_r)
                               )[0]
    #
    residual = r*u_tt - (r*u_rr + 2*u_r)
    loss_dom = residual 
    return loss_dom

def lossBCleft(r_bc,t_bc):
    u_bc    = model(r_bc,t_bc)
    # Derivatives
    u_bc_r  = torch.autograd.grad(outputs=u_bc, inputs=r_bc,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_bc)
                              )[0]
    #
    loss_bc = torch.pow(u_bc_r - 0.,2)
    return loss_bc

def lossBCright(r_bc,t_bc):
    u_bc    = model(r_bc,t_bc)
    # Derivatives
    u_bc_r  = torch.autograd.grad(outputs=u_bc, inputs=r_bc,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_bc)
                              )[0]
    u_bc_t  = torch.autograd.grad(outputs=u_bc, inputs=t_bc,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_bc)
                              )[0]
    #
    loss_bc = torch.pow((4./3.)*u_bc_t + u_bc_r + u_bc/r_bc - 0.,2)
    return loss_bc
#
def lossIC(r_ic,t_ic):
    u_ic     = model(r_ic,t_ic)
    # Derivatives
    u_ic_t   = torch.autograd.grad(outputs=u_ic, 
                              inputs=t_ic,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_ic)
                              )[0]
    #
    loss_ic  = torch.pow(u_ic - torch.exp(-torch.pow((40.*r_ic - 15.),2)),2)    
    loss_ic += torch.pow(u_ic_t - 0.,2)
    return loss_ic

def random_domain_points(R, T, n=8192):
    r = R*torch.rand(n, 1, device=device, requires_grad=True)
    t = T*torch.rand(n, 1, device=device, requires_grad=True)
    return r, t
def random_BC_points_L(R, T, n=512):
    r = 0*torch.ones((n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T*torch.rand(n, 1, device=device, requires_grad=True)
    return r, t
def random_BC_points_R(R, T, n=512):
    r = R*torch.ones((n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T*torch.rand(n, 1, device=device, requires_grad=True)
    return r, t
def random_IC_points(R, n=128):
    r = R*torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.zeros(n, 1, device=device, requires_grad=True)
    return r, t

# layer_sizes
layer_sizes = [2,64,64,64,1]
activation = nn.Tanh()
model = Model(layer_sizes,activation).to(device)


r,      t      = random_domain_points(L,T,n=TRAIN_DOM_POINTS)
r_bc  , t_bc   = random_BC_points_L(L,T,n=TRAIN_BC_POINTS)
r_bc_R, t_bc_R = random_BC_points_R(L,T,n=TRAIN_BC_POINTS)
r_ic,   t_ic   = random_IC_points(L,n=TRAIN_IC_POINTS)
#
fig = plt.figure(figsize=(7,6))
# Fontsize of evething inside the plot
plt.rcParams.update({'font.size': 16})
#
plt.plot(r.cpu().detach().numpy(),t.cpu().detach().numpy(),'o',ms=1)
plt.plot(r_bc.cpu().detach().numpy(),t_bc.cpu().detach().numpy(),'o',ms=1)
plt.plot(r_bc_R.cpu().detach().numpy(),t_bc_R.cpu().detach().numpy(),'o',ms=1)
plt.plot(r_ic.cpu().detach().numpy(),t_ic.cpu().detach().numpy(),'o',ms=1)
#
filename = os.path.join(save_dir, f'mesh_initial.png')
plt.savefig(filename, dpi=300, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()

filename = os.path.join(save_dir, f'initial_sampling_points')
np.savez(filename,
         r=r.cpu().detach().numpy(),
         t=t.cpu().detach().numpy(),
         r_bc_R=r_bc_R.cpu().detach().numpy(),
         t_bc_R=t_bc_R.cpu().detach().numpy(),
         r_bc=r_bc.cpu().detach().numpy(),
         t_bc=t_bc.cpu().detach().numpy(),
         r_ic=r_ic.cpu().detach().numpy(),
         t_ic=t_ic.cpu().detach().numpy()
        )

loss_dom_list  = []
loss_bc_L_list = []
loss_bc_R_list = []
loss_ic_list   = []
#
t0 = time()
# Initial training set
stop_criteria = 100.
best_achieved = 100.
adapt_step    = 0

#
optimizer = torch.optim.Adam(model.parameters(),
                            lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.8)  # Ensure not to use verbose=True

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    residual = lossRes(r, t)
    loss_dom = torch.mean(torch.pow(residual, 2))
    loss_bc_L = torch.mean(lossBCleft(r_bc, t_bc))
    loss_bc_R = torch.mean(lossBCright(r_bc_R, t_bc_R))
    loss_ic = torch.mean(lossIC(r_ic, t_ic))
    loss = loss_dom + GAMMA1 * (loss_bc_L + loss_bc_R) + GAMMA2 * loss_ic

    loss_dom_list.append(loss_dom.item())
    loss_bc_L_list.append(loss_bc_L.item())
    loss_bc_R_list.append(loss_bc_R.item())
    loss_ic_list.append(loss_ic.item())

    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()

    if epoch % 2000 == 0:
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f"Epoch: {epoch} - Loss: {loss.item():>1.3e} - Learning Rate: {current_lr:>1.3e}")

print('Computing time', (time() - t0) / 60, '[min]')
#
filename = os.path.join(save_dir, f'final_trained_model')
torch.save(model.state_dict(), filename)
#
filename = os.path.join(save_dir, f'training_losses')
np.savez(filename,
         loss_dom_list=loss_dom_list,
         loss_bc_R_list=loss_bc_R_list,
         loss_bc_L_list=loss_bc_L_list,
         loss_ic_list=loss_ic_list
        )

fig = plt.figure(figsize=(7,6))
# Fontsize of evething inside the plot
plt.rcParams.update({'font.size': 16})
#
plt.plot(r.cpu().detach().numpy(),t.cpu().detach().numpy(),'o',ms=1)
plt.plot(r_bc.cpu().detach().numpy(),t_bc.cpu().detach().numpy(),'o',ms=1)
plt.plot(r_bc_R.cpu().detach().numpy(),t_bc_R.cpu().detach().numpy(),'o',ms=1)
plt.plot(r_ic.cpu().detach().numpy(),t_ic.cpu().detach().numpy(),'o',ms=1)
#
filename = os.path.join(save_dir, f'mesh_final.png')
plt.savefig(filename, dpi=300, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()

fig = plt.figure(figsize=(7,6))
# Fontsize of evething inside the plot
plt.rcParams.update({'font.size': 16})
#
plt.semilogy(loss_ic_list, label='IC Loss')
plt.semilogy(loss_bc_R_list, label='Right BC Loss')
plt.semilogy(loss_bc_L_list, label='Left BC Loss')
plt.semilogy(loss_dom_list, label='Domain Loss')
plt.grid(True, which="both", ls="--")
plt.legend()
#
filename = os.path.join(save_dir, f'training_losses.png')
plt.savefig(filename, dpi=300, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()

x = torch.linspace(0, L, 1024).view(-1, 1)
x = x.to(device)

# Generate 30 time points uniformly spaced over [0, T]
time_points = np.linspace(0, T, 30)

for t_i in time_points:
    t = t_i * torch.ones_like(x)
    t = t.to(device)
    nn_sol = model(x, t).cpu().detach().numpy()
    
    plt.figure()
    plt.plot(x.cpu(), nn_sol, linewidth=2)
    plt.title(r'$t_i$:' + str(t_i))
    plt.xlim(0, L)
    
    filename = os.path.join(save_dir, f'pinns_sol_{t_i:.2f}.png')
    plt.savefig(filename, dpi=300, facecolor=None, edgecolor=None,
                orientation='portrait', format='png', transparent=True, 
                bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.close()

  # CUDA_VISIBLE_DEVICES=1 python spherical_non_wave_eq.py > log_spherical_non_wave_eq_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &