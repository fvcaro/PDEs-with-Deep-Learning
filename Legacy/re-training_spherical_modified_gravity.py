import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import seaborn as sns

sns.set_style("whitegrid")

print('torch version:', torch.__version__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print('device:', device)

R = 40
T = 25
# v = 1
A = 1
delta = 1
r_0 = 15
u_0 = 0
#
GAMMA1 = 10.
GAMMA2 = 10.
#
SIGMA=1.
# GAMMA=1.e-6 # small perturbation
# GAMMA=1.e-5 # bigger perturbation
# GAMMA=1.e-4 # extra perturbation
GAMMA=0. # just wave_eq

LEARNING_RATE = 0.001

STOP_CRITERIA = 0.0001
EPOCHS = 100000

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.layer01 = torch.nn.Linear(2,20)
        self.layer02 = torch.nn.Linear(20,50)
        self.layer03 = torch.nn.Linear(50,50)
        self.layer04 = torch.nn.Linear(50,50)
        self.layer05 = torch.nn.Linear(50,20)
        self.layer06 = torch.nn.Linear(20,1)
    
    def forward(self,x,t):
        inputs      = torch.cat([x,t], axis=1)
        out_layer01 = torch.tanh(self.layer01(inputs))
        out_layer02 = torch.tanh(self.layer02(out_layer01))
        out_layer03 = torch.tanh(self.layer03(out_layer02))
        out_layer04 = torch.tanh(self.layer04(out_layer03))
        out_layer05 = torch.tanh(self.layer05(out_layer04))
        out_layer06 = self.layer06(out_layer05)
        output      = out_layer06
        return output

def K(X, sigma=1., gamma=1.):
    K = -sigma/2.*X - gamma/8.*X**3
    return K

def der_K_X(X, sigma=1., gamma=1.):
    K_X = -sigma/2. - 3*gamma/8.*X**2
    return K_X

def der_K_XX(X, sigma=1., gamma=1.):
    K_XX = - 3*gamma/4.*X
    return K_XX

def loss_1(r, t):
    u = model(r, t)
    # Derivatives
    u_t  = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_tr  = torch.autograd.grad(u_t, r, create_graph=True, grad_outputs=torch.ones_like(u_t))[0]
    u_tt = torch.autograd.grad(u_t, t, create_graph=True, grad_outputs=torch.ones_like(u_t))[0]
    
    u_r  = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_rt  = torch.autograd.grad(u_r, t, create_graph=True, grad_outputs=torch.ones_like(u_r))[0]
    u_rr = torch.autograd.grad(u_r, r, create_graph=True, grad_outputs=torch.ones_like(u_r))[0]

    X = (u_r)**2 - (u_t)**2

    K_2_XX = der_K_XX(X, sigma=SIGMA, gamma=GAMMA)
    K_1_X  = der_K_X(X, sigma=SIGMA, gamma=GAMMA)

    # Residual
    residual = 2*r*K_2_XX*u_rr*(u_r)**2 - 4*r*K_2_XX*u_t*u_tr*u_r + 2*r*K_2_XX*(u_t)**2*u_tt + K_1_X*(2*u_r + r*u_rr - r*u_tt)

    return residual

def loss_2_L(r, t):
    u = model(r, t)
    # Derivatives
    u_r = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]

    # Left Boundary Condition (u_r(t, 0) = 0)
    loss_left_bc = u_r**2

    loss_bc = loss_left_bc

    return loss_bc

def loss_2_R(r, t):
    u = model(r, t)
    # Derivatives
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_r = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]

    # Right Boundary Condition
    loss_right_bc = (u_t + u_r + u/r)**2

    loss_bc = loss_right_bc

    return loss_bc

def loss_3(r_ic, t_ic):
    u_ic = model(r_ic, t_ic)
    # Derivative with respect to time
    u_ic_t = torch.autograd.grad(outputs=u_ic, inputs=t_ic, create_graph=True, grad_outputs=torch.ones_like(u_ic))[0]
    
    # Initial condition losses
    ic_condition_1 = u_ic - A * torch.exp(-((r_ic - r_0) / delta)**2)
    ic_condition_2 = u_ic_t
    
    # Combine the losses
    loss_ic = ic_condition_1**2 + ic_condition_2**2
    return loss_ic

torch.manual_seed(42)
model = Model().to(device)

model.load_state_dict(torch.load('trained_model_gpu_spherical',map_location=torch.device(device)))
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Update the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=0.0001, last_epoch=-1)
# scheduler = StepLR(optimizer, step_size=2000, gamma=1., verbose=False)  # Learning rate scheduler

# Define a directory to save the figures
save_dir = 'Re-Tr'
os.makedirs(save_dir, exist_ok=True)
#
aux  = np.load('adaptive_sampling_points.npz')
#
r      = aux['r']
t      = aux['t']
r_bc_L = aux['r_bc_L']
t_bc_L = aux['t_bc_L']
r_bc_R = aux['r_bc_R']
t_bc_R = aux['t_bc_R']
r_ic   = aux['r_ic']
t_ic   = aux['t_ic']
#
r      = torch.tensor(r).view(-1,1)
t      = torch.tensor(t).view(-1,1)
r_bc_L = torch.tensor(r_bc_L).view(-1,1)
t_bc_L = torch.tensor(t_bc_L).view(-1,1)
r_bc_R = torch.tensor(r_bc_R).view(-1,1)
t_bc_R = torch.tensor(t_bc_R).view(-1,1)
r_ic   = torch.tensor(r_ic).view(-1,1)
t_ic   = torch.tensor(t_ic).view(-1,1)
#
r.requires_grad_(True)
t.requires_grad_(True)
r_bc_L.requires_grad_(True)
t_bc_L.requires_grad_(True)
r_bc_R.requires_grad_(True)
t_bc_R.requires_grad_(True)
r_ic.requires_grad_(True)
t_ic.requires_grad_(True)
#
r      = r.to(device)
t      = t.to(device)
r_bc_L = r_bc_L.to(device)
t_bc_L = t_bc_L.to(device)
r_bc_R = r_bc_R.to(device)
t_bc_R = t_bc_R.to(device)
r_ic   = r_ic.to(device)
t_ic   = t_ic.to(device)
#
loss_dom_list  = []
loss_bc_L_list = []
loss_bc_R_list = []
loss_ic_list   = []

t0 = time()

for epoch in range(EPOCHS):
    # Track epochs
    #if (epoch%(epochs/10)==0):
    #    print('epoch:',epoch)
    optimizer.zero_grad() # to make the gradients zero
    # RESIDUAL ################################################################ 
    residual = loss_1(r, t)
    residual_r = torch.autograd.grad(residual, r, create_graph=True, grad_outputs=torch.ones_like(residual))[0]
    residual_t = torch.autograd.grad(residual, t, create_graph=True, grad_outputs=torch.ones_like(residual))[0]
    loss_dom = torch.mean(residual**2 + residual_r**2 + residual_t**2)
    # BC ######################################################################
    loss_bc_L  =  torch.mean(loss_2_L(r_bc_L,t_bc_L))
    loss_bc_R  =  torch.mean(loss_2_R(r_bc_R,t_bc_R))
    # IC ######################################################################
    loss_ic  = torch.mean(loss_3(r_ic,t_ic))
    # LOSS ####################################################################
    loss = loss_dom + GAMMA1*loss_bc_L + GAMMA1*loss_bc_R + GAMMA2*loss_ic # + loss_dom_t 
    stop_criteria = loss.item()
    # Calculate and append individual losses to their respective lists
    loss_dom_list.append(loss_dom.item())
    loss_bc_L_list.append(loss_bc_L.item())
    loss_bc_R_list.append(loss_bc_R.item())
    loss_ic_list.append(loss_ic.item())

    loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
    optimizer.step() # 
    scheduler.step()  # Update learning rate
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} - Loss: {loss.item():>1.6e} - Learning Rate: {scheduler.get_last_lr()[0]:>1.6e}")
    # Check if stop criteria are met
    if stop_criteria < STOP_CRITERIA:
        print(f"Iteration {iteration}: Stop criteria met. Loss: {stop_criteria}")
        break
    #
print('computing time',(time() - t0)/60,'[min]')  

torch.save(model.state_dict(), 'trained_model_gpu_spherical')

# Plotting individual losses
plt.figure(figsize=(10, 6))
plt.semilogy(loss_dom_list, label='Domain Loss')
plt.semilogy(loss_bc_L_list, label='Left BC Loss')
plt.semilogy(loss_bc_R_list, label='Right BC Loss')
plt.semilogy(loss_ic_list, label='Initial Condition Loss')
plt.legend()
plt.title('Individual Losses Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True, which="both", ls="--")
save_filename = os.path.join(save_dir, 'individual_losses.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

np.save('loss_adaptive_sampling_gpu', loss_dom_list)


x = torch.linspace(-0,R,256).view(-1,1)

for t_i in np.linspace(0, T, 11):
    t = t_i * torch.ones_like(x)
    # Move x and t to the same device as the model
    t = t.to(device)
    x = x.to(device)
    nn_sol = model(x, t).cpu().detach().numpy()

    # Plot the figure
    plt.plot(x.cpu().numpy(), nn_sol, label=f't = {t_i}')
    save_filename = os.path.join(save_dir, f'figure_t_{t_i}.png')
    plt.xlabel('x-axis')  # Add your x-axis label
    plt.ylabel('y-axis')  # Add your y-axis label
    plt.legend()
    plt.savefig(save_filename, dpi=300)
    plt.close()  # Close the figure to release resources

from matplotlib.animation import FuncAnimation
# plt.style.use('seaborn-pastel')

fig = plt.figure(figsize=(8,6))
ax = plt.axes(xlim=(0, R), ylim=(-2., 2.))
line2, = ax.plot([], [], 
                 color='tab:blue',
                 lw=2,
                 linestyle='--',
                 label='pinn sol'
                )
ax.text(0.1, 0.9, 
        "t = ", 
        bbox={'facecolor': 'white',
              'alpha': 0.5, 
              'pad': 5},
        transform=ax.transAxes, 
        ha="center")
#
ax.legend()
# #
def init():
    #
    line2.set_data([], [])
    return line2,
def animate(i):
    #####################################################
    ax.text(0.1, 0.9, 
            "t= %d" % i,
            bbox={'facecolor': 'white', 
                  'alpha': 0.5, 
                  'pad': 5},
            transform=ax.transAxes, 
            ha="center")
#     #####################################################
#     x_np = np.linspace(-R,R,512)
    t = i
#     y_np = np.exp(-A*((x_np - x0) - c*t)**2)/2 + np.exp(-A*((x_np - x0) + c*t)**2)/2
#     #####################################################
    x_tr = torch.linspace(0,R,512).view(-1,1)
    x_tr = x_tr.to(device)
#     #
    t_tr = t*torch.ones_like(x_tr)
    t_tr = t_tr.to(device)
    y_tr = model(x_tr,t_tr).cpu().detach().numpy()
#     #
    line2.set_data(x_tr.cpu().detach().numpy(), y_tr)
    return line2,

anim = FuncAnimation(fig, animate, 
                     init_func=init,
                     frames=np.linspace(0, 25, 51), 
                     blit=True
                    )
save_filename = os.path.join(save_dir, 'final_wave_animation_gpu.gif')
anim.save(save_filename, writer='imagemagick')

# python re-training_spherical_modified_gravity.py > log_re-training_modified_gravity_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python re-training_spherical_modified_gravity.py > log_re-training_modified_gravity_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python re-training_spherical_modified_gravity.py > log_re-training_modified_gravity_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &