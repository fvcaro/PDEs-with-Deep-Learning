import torch
print('torch version:', torch.__version__)
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from time import time
import os
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)  # Set random seed for reproducibility
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
# GAMMA=1.e-4 # EXTRA perturbation
GAMMA=0. # just wave_eq
#
TRAIN_DOM_POINTS = 16384
TRAIN_BC_POINTS  = 64
TRAIN_IC_POINTS  = 32

VALID_DOM_POINTS = 65536
VALID_BC_POINTS  = 256
VALID_IC_POINTS  = 128

DOM_NEW_POINTS = 128
BC_NEW_POINTS  = 16
IC_NEW_POINTS  = 16

LEARNING_RATE = 0.0005

STOP_CRITERIA = 0.0001
ITER_MAX = 100 # Set a reasonable maximum number of iterations
EPOCHS = 100

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

def K(X, sigma=1., gamma=1.):
    K = -(sigma/2.)*X - (gamma/8.)*X**3
    return K

def der_K_X(X, sigma=1., gamma=1.):
    K_X = -sigma/2. - 3*(gamma/8.)*X**2
    return K_X

def der_K_XX(X, sigma=1., gamma=1.):
    K_XX = - (3*gamma/4.)*X
    return K_XX

def loss_domain(r, t):
    u = model(r, t)
    # Pi and Phi
    Pi = u[:,0].view(-1,1)
    Phi = u[:,1].view(-1,1)
    # Derivatives of the components
    Pi_t  = torch.autograd.grad(Pi, t,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Pi)
                               )[0]
    #
    Pi_r  = torch.autograd.grad(Pi, r,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Pi)
                               )[0]
    #
    Phi_t  = torch.autograd.grad(Phi, t,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Phi)
                               )[0]
    #
    Phi_r  = torch.autograd.grad(Phi, r,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Phi)
                               )[0]
    #
    X = Phi**2 - Pi**2
    #
    K_XX = der_K_XX(X, sigma=SIGMA, gamma=GAMMA)
    K_X  = der_K_X(X, sigma=SIGMA, gamma=GAMMA)
    # Residuals
    residual_1 = (2*K_XX*Pi**2 - K_X)*Pi_t + (2*K_XX*Phi**2 + K_X)*Phi_r + (2*K_X)/r*(Phi) - 4*K_XX*Phi*Pi*Pi_r
    residual_2 = Phi_t - Pi_r
    return residual_1, residual_2

def loss_L_BC(r, t):
    u_bc = model(r,t)
    # Phi
    Phi = u_bc[:,1].view(-1,1)
    # Phi Left BC (Phi(t, 0) = 0)
    Phi_left_bc = Phi**2

    return Phi_left_bc

def loss_R_BC(r, t):
    u_bc = model(r,t)
    # Pi and Phi
    Pi = u_bc[:,0].view(-1,1)
    Phi = u_bc[:,1].view(-1,1)
    # Derivatives of the components
    Pi_t  = torch.autograd.grad(Pi, t,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Pi)
                               )[0]
    Pi_r  = torch.autograd.grad(Pi, r,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Pi)
                               )[0]
    Phi_t  = torch.autograd.grad(Phi, t,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Phi)
                               )[0]
    Phi_r  = torch.autograd.grad(Phi, r,
                                create_graph=True,
                                grad_outputs=torch.ones_like(Phi)
                               )[0]
    # Pi Right BC
    Pi_right_bc = (Pi_t + Pi_r + Pi/r)**2
    # Phi Right BC
    Phi_right_bc = (Phi_t + Phi_r + Phi/r)**2
    #
    return Pi_right_bc, Phi_right_bc

def loss_IC(r_ic, t_ic):
    u_ic = model(r_ic, t_ic)
    # Pi and Phi
    Pi = u_ic[:,0].view(-1,1)
    Phi = u_ic[:,1].view(-1,1)
    # ICs
    Pi_IC = (Pi - 0.)**2
    # A * torch.exp(-((r_ic - r_0) / delta)**2)
    Phi_IC = (Phi + 2*A*(r_ic - r_0)/(delta**2)*torch.exp(-((r_ic - r_0) / delta)**2))**2

    return Pi_IC, Phi_IC

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

# layer_sizes = [2,128,128,128,1]
layer_sizes = [2,64,64,64,2]
activation = nn.Tanh()
model = Model(layer_sizes,activation).to(device)

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
# print("Sizes: r={}, t={}, r_bc_L={}, t_bc_L={}, r_bc_R={}, t_bc_R={}, r_ic={}, t_ic={}".format(
    # r.size(), t.size(), r_bc_L.size(), t_bc_L.size(), r_bc_R.size(), t_bc_R.size(), r_ic.size(), t_ic.size()))
plt.plot(r.detach().cpu().numpy(),t.detach().cpu().numpy(),'o',ms=1)
plt.plot(r_bc_L.detach().cpu().numpy(),t_bc_L.detach().cpu().numpy(),'o', ms=1)
plt.plot(r_bc_R.detach().cpu().numpy(),t_bc_R.detach().cpu().numpy(),'o', ms=1)
plt.plot(r_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'points.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

loss_dom_1_list  = []
loss_dom_2_list  = []
loss_L_BC_Phi_list = []
# loss_bc_R_list = []
# loss_ic_list   = []

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
        residual_1, residual_2 = loss_domain(r, t)
        #
        residual_1_r = torch.autograd.grad(residual_1, r, create_graph=True, grad_outputs=torch.ones_like(residual_1))[0]
        residual_2_r = torch.autograd.grad(residual_2, r, create_graph=True, grad_outputs=torch.ones_like(residual_2))[0]
        #
        residual_1_t = torch.autograd.grad(residual_1, t, create_graph=True, grad_outputs=torch.ones_like(residual_1))[0]
        residual_2_t = torch.autograd.grad(residual_2, t, create_graph=True, grad_outputs=torch.ones_like(residual_2))[0]
        #
        loss_dom_1 = torch.mean(residual_1**2 + residual_1_r**2)
        loss_dom_2 = torch.mean(residual_2**2 + residual_2_r**2)
        # L_BC ######################################################################
        loss_L_BC_Phi = loss_L_BC(r_bc_L,t_bc_L)
        loss_L_BC_Phi = torch.mean(loss_L_BC_Phi)
        # R_BC ######################################################################
        loss_R_BC_Pi, loss_R_BC_Phi  =  loss_R_BC(r_bc_R,t_bc_R)
        loss_R_BC_Pi = torch.mean(loss_R_BC_Pi)
        loss_R_BC_Phi = torch.mean(loss_R_BC_Phi)
        # IC ######################################################################
        loss_IC_Pi, loss_IC_Phi  = loss_IC(r_ic,t_ic)
        loss_IC_Pi = torch.mean(loss_IC_Pi)
        loss_IC_Phi = torch.mean(loss_IC_Phi)
        # LOSS ####################################################################
        loss_Pi = loss_dom_1  + GAMMA1*loss_R_BC_Pi + GAMMA2*loss_IC_Pi
        loss_Phi = loss_dom_2 + GAMMA1*loss_L_BC_Phi + GAMMA1*loss_R_BC_Phi + GAMMA2*loss_IC_Phi
        loss = loss_Pi + loss_Phi
        # Calculate and append individual losses to their respective lists
        loss_dom_1_list.append(loss_dom_1.item())
        loss_dom_2_list.append(loss_dom_2.item())
        loss_L_BC_Phi_list.append(loss_L_BC_Phi.item())
        # loss_bc_R_list.append(loss_bc_R.item())
        # loss_ic_list.append(loss_ic.item())

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # 
        scheduler.step()  # Update learning rate
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>1.6e} - Learning Rate: {scheduler.get_last_lr()[0]:>1.6e}")
    # Adative re-sampling ######################################################################
    r_,t_                    = random_domain_points(R,T,n=VALID_DOM_POINTS)
    r_bc_L_,t_bc_L_          = random_BC_points_L(R,T,n=VALID_BC_POINTS)
    r_bc_R_,t_bc_R_          = random_BC_points_R(R,T,n=VALID_BC_POINTS)
    r_ic_, t_ic_             = random_IC_points(R,n=VALID_IC_POINTS)
    # RESIDUAL ################################################################ 
    residual_1_, residual_2_ = loss_domain(r_, t_)
    residual_ = residual_1_ + residual_2_
    residual_1_r_ = torch.autograd.grad(residual_1_, r_, create_graph=True, grad_outputs=torch.ones_like(residual_1_))[0]
    residual_2_r_ = torch.autograd.grad(residual_2_, r_, create_graph=True, grad_outputs=torch.ones_like(residual_2_))[0]
    residual_r_ = residual_1_r_ + residual_2_r_
    loss_dom_aux  = residual_**2 + residual_r_**2
    # loss_L_BC_aux ######################################################################
    loss_L_BC_Phi_aux = loss_L_BC(r_bc_L_,t_bc_L_)
    loss_bc_L_aux = loss_L_BC_Phi_aux
    # loss_R_BC_aux ######################################################################
    loss_R_BC_Pi_aux, loss_R_BC_Phi_aux  =  loss_R_BC(r_bc_R_,t_bc_R_)
    loss_bc_R_aux = loss_R_BC_Pi_aux + loss_R_BC_Phi_aux
    # IC ######################################################################
    loss_IC_Pi, loss_IC_Phi  = loss_IC(r_ic_,t_ic_)
    loss_ic_aux = loss_IC_Pi + loss_IC_Phi
    # INDICATORS ################################################################
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
plt.plot(r_bc_L.detach().cpu().numpy(),t_bc_L.detach().cpu().numpy(),'o', ms=1)
plt.plot(r_bc_R.detach().cpu().numpy(),t_bc_R.detach().cpu().numpy(),'o', ms=1)
plt.plot(r_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'new_points.png')
plt.savefig(save_filename, dpi=600, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()  # Close the figure to release resources

#np.savez('adaptive_sampling_points',x=r.detach().numpy(),t=t.detach().numpy())

# Plotting individual losses
plt.figure(figsize=(10, 6))
plt.semilogy(loss_dom_1_list, label='Domain Loss Pi')
plt.semilogy(loss_dom_2_list, label='Domain Loss Phi')
plt.semilogy(loss_L_BC_Phi_list, label='Left Phi BC Loss')
# plt.semilogy(loss_ic_list, label='Initial Condition Loss')
plt.legend()
plt.title('Individual Losses Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True, which="both", ls="--")
save_filename = os.path.join(save_dir, 'individual_losses.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

np.save('loss_adaptive_sampling_gpu', loss_dom_1_list)

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

# ANIMATION ####################################################################
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(xlim=(0, R), ylim=(-2., 2.))
line2, = ax.plot([], [], color='tab:blue', lw=2, linestyle='--', label='pinn sol')
# ax.text(0.1, 0.9, "t = ", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha="center")
ax.legend()

# Add gridlines
plt.grid(True, which="both", ls="--")

# Pre-calculate predictions for all frames
x_tr = torch.linspace(0, R, 512).view(-1, 1).to(device)
t_values = torch.linspace(0, 25, 51).tolist()
y_preds = [model(x_tr, t * torch.ones_like(x_tr)).cpu().detach().numpy() for t in t_values]

# Initialize text outside animation loop
text_template = ax.text(0.1, 0.9, "", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5}, transform=ax.transAxes, ha="center")

def init():
    line2.set_data([], [])
    return line2,

def animate(i):
    # Update text with current time
    text_template.set_text("t = %d" % t_values[i])
    
    # Update line with pre-calculated predictions
    line2.set_data(x_tr.cpu().detach().numpy(), y_preds[i])
    return line2,

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(t_values), blit=True)

# Save animation using Pillow writer
save_filename = os.path.join(save_dir, 'final_wave_animation_gpu.gif')
anim.save(save_filename, writer='pillow')

# CUDA_VISIBLE_DEVICES=0 python spherical_first_order.py > log_modified_gravity_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python spherical_first_order.py > log_modified_gravity_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &