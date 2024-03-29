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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

R = 40
T = 20
v = 1
A = 1
delta = 1
r_0 = 15
u_0 = 0
#
GAMMA1 = 10.
GAMMA2 = 10.
#
TRAIN_DOM_POINTS = 2048
TRAIN_BC_POINTS  = 64
TRAIN_IC_POINTS  = 32

VALID_DOM_POINTS = 16384
VALID_BC_POINTS  = 256
VALID_IC_POINTS  = 128

DOM_NEW_POINTS = 128
BC_NEW_POINTS  = 16
IC_NEW_POINTS  = 16

LEARNING_RATE = 0.001

STOP_CRITERIA = 0.0001
ITER_MAX = 100  # Set a reasonable maximum number of iterations
EPOCHS = 10000

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

def loss_1(r, t):
    u = model(r, t)
    # Derivatives
    u_t  = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_tt = torch.autograd.grad(u_t, t, create_graph=True, grad_outputs=torch.ones_like(u_t))[0]
    u_ttr = torch.autograd.grad(u_tt, r, create_graph=True, grad_outputs=torch.ones_like(u_tt))[0]
    u_ttt = torch.autograd.grad(u_tt, t, create_graph=True, grad_outputs=torch.ones_like(u_tt))[0]
    u_r  = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_rt  = torch.autograd.grad(u_r, t, create_graph=True, grad_outputs=torch.ones_like(u_r))[0]
    u_rr = torch.autograd.grad(u_r, r, create_graph=True, grad_outputs=torch.ones_like(u_r))[0]
    u_rrt = torch.autograd.grad(u_rr, t, create_graph=True, grad_outputs=torch.ones_like(u_rr))[0]
    u_rrr = torch.autograd.grad(u_rr, r, create_graph=True, grad_outputs=torch.ones_like(u_rr))[0]

    # residual = u_tt - v**2 * (u_rr + (2/r) * u_r)
    # Modified residual
    residual = r * u_tt - v**2 * (r * u_rr + 2 * u_r)
    residual_t = r * u_ttt - v**2 * (r * u_rrt + 2 * u_rt)
    residual_r = u_tt + r * u_ttr - v**2 * (u_rr + r * u_rrr + 2 * u_rr)

    return residual, residual_t, residual_r

def loss_2_L(r, t):
    u = model(r, t)
    # Derivatives
    u_r = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]

    # Left Boundary Condition (u_r(0, t) = 0)
    loss_left_bc = u_r**2

    loss_bc = loss_left_bc

    return loss_bc

# def loss_2_L(r, t):
#     u = model(r, t)
    
#     # Left Boundary Condition (u(0, t) = 0)
#     loss_left_bc = u**2

#     loss_bc = loss_left_bc

#     return loss_bc

def loss_2_R(r, t):
    u = model(r, t)
    # Derivatives
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_r = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]

    # Right Boundary Condition
    loss_right_bc = (u_t + (v/r) * u_r + v * (u - u_0)/r)**2

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

def random_domain_points(R, T, n=8192):
    r = R * torch.rand(n, 1, device=device, requires_grad=True)
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return r, t
def random_BC_points_L(R, T, n=512):
    r = 0 * torch.ones((n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return r, t
def random_BC_points_R(R, T, n=512):
    r = R * torch.ones((n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return r, t
def random_IC_points(R, n=128):
    r = R * torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.zeros(n, 1, device=device, requires_grad=True)
    return r, t

torch.manual_seed(42)
model = Model().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Update the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=0.0001, last_epoch=-1)
# scheduler = StepLR(optimizer, step_size=2000, gamma=1., verbose=False)  # Learning rate scheduler

# Define a directory to save the figures
save_dir = 'Figs'
os.makedirs(save_dir, exist_ok=True)

r,t            = random_domain_points(R,T,n=TRAIN_DOM_POINTS)
r_bc_L, t_bc_L = random_BC_points_L(R,T,n=TRAIN_BC_POINTS)
r_bc_R, t_bc_R = random_BC_points_R(R,T,n=TRAIN_BC_POINTS)
r_ic, t_ic     = random_IC_points(R,n=TRAIN_IC_POINTS)
print("Sizes: r={}, t={}, r_bc_L={}, t_bc_L={}, r_bc_R={}, t_bc_R={}, r_ic={}, t_ic={}".format(
    r.size(), t.size(), r_bc_L.size(), t_bc_L.size(), r_bc_R.size(), t_bc_R.size(), r_ic.size(), t_ic.size()))
plt.plot(r.detach().cpu().numpy(),t.detach().cpu().numpy(),'o',ms=1)
plt.plot(r_bc_L.detach().cpu().numpy(),t_bc_L.detach().cpu().numpy(),'o')
plt.plot(r_bc_R.detach().cpu().numpy(),t_bc_R.detach().cpu().numpy(),'o')
plt.plot(r_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'points.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

loss_dom_list  = []
loss_bc_L_list = []
loss_bc_R_list = []
loss_ic_list   = []

t0 = time()

# Initial training set
stop_criteria = 1.
best_stop_criteria = 1.
iteration = 0

while stop_criteria > STOP_CRITERIA and iteration < ITER_MAX:
    for epoch in range(EPOCHS):
        # Track epochs
        #if (epoch%(epochs/10)==0):
        #    print('epoch:',epoch)
        optimizer.zero_grad() # to make the gradients zero
        # RESIDUAL ################################################################ 
        residual, residual_t, residual_r = loss_1(r, t)
        loss_dom = torch.mean(residual**2 + residual_r**2) # residual_t**2
        # BC ######################################################################
        loss_bc_L  =  torch.mean(loss_2_L(r_bc_L,t_bc_L))
        loss_bc_R  =  torch.mean(loss_2_R(r_bc_R,t_bc_R))
        # IC ######################################################################
        loss_ic  = torch.mean(loss_3(r_ic,t_ic))
        # LOSS ####################################################################
        loss = loss_dom + GAMMA1*loss_bc_L + GAMMA1*loss_bc_R + GAMMA2*loss_ic # + loss_dom_t 
        # Calculate and append individual losses to their respective lists
        loss_dom_list.append(loss_dom.cpu().detach().numpy())
        loss_bc_L_list.append(loss_bc_L.cpu().detach().numpy())
        loss_bc_R_list.append(loss_bc_R.cpu().detach().numpy())
        loss_ic_list.append(loss_ic.cpu().detach().numpy())

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # 
        scheduler.step()  # Update learning rate
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>1.6e} - Learning Rate: {scheduler.get_last_lr()[0]:>1.6e}")
    # Adative sample step 
    r_,t_                  = random_domain_points(R,T,n=VALID_DOM_POINTS)

    r_bc_L_,t_bc_L_        = random_BC_points_L(R,T,n=VALID_BC_POINTS)
    r_bc_R_,t_bc_R_        = random_BC_points_R(R,T,n=VALID_BC_POINTS)

    r_ic_, t_ic_  = random_IC_points(R,n=VALID_IC_POINTS)
    #
    residual_, residual_t_, residual_r_ = loss_1(r_, t_)
    loss_dom_aux  = residual_**2
    loss_bc_L_aux = loss_2_L(r_bc_L_,t_bc_L_)
    loss_bc_R_aux = loss_2_R(r_bc_R_,t_bc_R_)
    loss_ic_aux   = loss_3(r_ic_,t_ic_)   
    #
    idx_dom  = torch.where(loss_dom_aux >= loss_dom_aux.sort(0)[0][-DOM_NEW_POINTS])[0]
    idx_bc_L = torch.where(loss_bc_L_aux >= loss_bc_L_aux.sort(0)[0][-BC_NEW_POINTS])[0]
    idx_bc_R = torch.where(loss_bc_R_aux >= loss_bc_R_aux.sort(0)[0][-BC_NEW_POINTS])[0]
    idx_ic   = torch.where(loss_ic_aux >= loss_ic_aux.sort(0)[0][-IC_NEW_POINTS])[0]
    #
    r_aux = r_[idx_dom].view(-1, 1) if len(idx_dom) > 0 else torch.tensor([]).view(0, 1)
    t_aux = t_[idx_dom].view(-1, 1) if len(idx_dom) > 0 else torch.tensor([]).view(0, 1)
    r = torch.cat((r,r_aux),0)
    t = torch.cat((t,t_aux),0)
    #
    r_bc_L_aux = r_bc_L_[idx_bc_L].view(-1,1)
    t_bc_L_aux = t_bc_L_[idx_bc_L].view(-1,1)
    r_bc_L     = torch.cat((r_bc_L,r_bc_L_aux),0)
    t_bc_L     = torch.cat((t_bc_L,t_bc_L_aux),0)
    #
    r_bc_R_aux = r_bc_R_[idx_bc_R].view(-1,1)
    t_bc_R_aux = t_bc_R_[idx_bc_R].view(-1,1)
    r_bc_R     = torch.cat((r_bc_R,r_bc_R_aux),0)
    t_bc_R     = torch.cat((t_bc_R,t_bc_R_aux),0)
    #
    r_ic_aux = r_ic_[idx_ic].view(-1,1)
    t_ic_aux = t_ic_[idx_ic].view(-1,1)
    r_ic     = torch.cat((r_ic,r_ic_aux),0)
    t_ic     = torch.cat((t_ic,t_ic_aux),0)
    #
    stop_criteria = loss_dom_aux.sort(0)[0][-1].cpu().detach().numpy()[0]
    if stop_criteria < best_stop_criteria:
        best_stop_criteria = stop_criteria
        torch.save(model.state_dict(), 'best_trained_model_gpu_spherical')
        np.savez('best_adaptive_sampling_points',r=r.cpu().detach().numpy(),
                                    t=t.cpu().detach().numpy(),
                                    r_bc_R=r_bc_R.cpu().detach().numpy(),
                                    t_bc_R=t_bc_R.cpu().detach().numpy(),
                                    r_bc_L=r_bc_L.cpu().detach().numpy(),
                                    t_bc_L=t_bc_L.cpu().detach().numpy(),
                                    r_ic=r_ic.cpu().detach().numpy(),
                                    t_ic=t_ic.cpu().detach().numpy()
                                    )
    # Check if stop criteria are met
    if stop_criteria > 0.0001:
        iteration += 1
        print(f"Iteration {iteration}: Best criteria        . Loss: {best_stop_criteria}")
        print(f"Iteration {iteration}: Stop criteria not met. Loss: {stop_criteria}")
    #
print('computing time',(time() - t0)/60,'[min]')  

loss_dom_aux.sort(0)[0][-1].cpu().detach().numpy()[0]

torch.save(model.state_dict(), 'trained_model_gpu_spherical')
np.savez('adaptive_sampling_points',r=r.cpu().detach().numpy(),
                                    t=t.cpu().detach().numpy(),
                                    r_bc_R=r_bc_R.cpu().detach().numpy(),
                                    t_bc_R=t_bc_R.cpu().detach().numpy(),
                                    r_bc_L=r_bc_L.cpu().detach().numpy(),
                                    t_bc_L=t_bc_L.cpu().detach().numpy(),
                                    r_ic=r_ic.cpu().detach().numpy(),
                                    t_ic=t_ic.cpu().detach().numpy()
                                    )
plt.figure()
plt.plot(r.detach().cpu().numpy(),t.detach().cpu().numpy(),'o',ms=1)
plt.plot(r_bc_L.detach().cpu().numpy(),t_bc_L.detach().cpu().numpy(),'o')
plt.plot(r_bc_R.detach().cpu().numpy(),t_bc_R.detach().cpu().numpy(),'o')
plt.plot(r_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'new_points.png')
plt.savefig(save_filename, dpi=600, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()  # Close the figure to release resources

#np.savez('adaptive_sampling_points',x=r.detach().numpy(),t=t.detach().numpy())

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
ax = plt.axes(xlim=(0, R), ylim=(-1., 1.))
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
    x_tr = torch.linspace(-R,R,512).view(-1,1)
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
                     frames=np.linspace(0, 20, 41), 
                     blit=True
                    )
save_filename = os.path.join(save_dir, 'final_wave_animation_gpu.gif')
anim.save(save_filename, writer='imagemagick')

# python spherical_coord_wave_eq.py > log_spherical_coord_wave_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python spherical_coord_wave_eq.py > log_spherical_coord_wave_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &