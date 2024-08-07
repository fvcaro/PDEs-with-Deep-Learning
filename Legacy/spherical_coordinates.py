import torch
print('torch version:', torch.__version__)
import torch.nn as nn
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
T = 30
v = 1 # c velocity parameter
A = 1
delta = 1
r_0 = 15
u_0 = 0
#
GAMMA1 = 10.
GAMMA2 = 10.
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

LEARNING_RATE = 0.001

STOP_CRITERIA = 0.0001
ITER_MAX = 100 # Set a reasonable maximum number of iterations
EPOCHS = 10000

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
                # Kaiming initialization
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

def loss_domain(r, t):
    u = model(r, t)
    # Derivatives
    u_t  = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_tr  = torch.autograd.grad(u_t, r, create_graph=True, grad_outputs=torch.ones_like(u_t))[0]
    u_tt = torch.autograd.grad(u_t, t, create_graph=True, grad_outputs=torch.ones_like(u_t))[0]
    
    u_r  = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_rr = torch.autograd.grad(u_r, r, create_graph=True, grad_outputs=torch.ones_like(u_r))[0]

    # Residual:
    residual = r*u_tt-v**2*(r*u_rr+2*u_r)

    return residual

def loss_L_BC(r, t):
    u = model(r, t)
    # Derivatives
    u_r = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    # Left Boundary Condition (u_r(t, 0) = 0)
    loss_left_bc = (u_r - 0.)**2
    return loss_left_bc

def loss_R_BC(r, t):
    u = model(r, t)
    # Derivatives
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_r = torch.autograd.grad(u, r, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    # Right Boundary Condition
    loss_right_bc = (u_t + v*u_r + (v/r)*u)**2

    return loss_right_bc

def loss_IC(r_ic, t_ic):
    u_ic = model(r_ic, t_ic)
    # Derivative with respect to time
    u_ic_t = torch.autograd.grad(outputs=u_ic, inputs=t_ic, create_graph=True, grad_outputs=torch.ones_like(u_ic))[0]
    # Initial condition losses
    loss_ic = (u_ic - A*torch.exp(-((r_ic - r_0) / delta)**2))**2 + (u_ic_t - 0.)**2
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

# layer_sizes = [2,128,128,128,1]
layer_sizes = [2,64,64,64,1]
activation = nn.Tanh()
model = Model(layer_sizes,activation).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define a directory to save the figures
save_dir = 'Figs_spherical_coords'
os.makedirs(save_dir, exist_ok=True)

r,t            = random_domain_points(R,T,n=TRAIN_DOM_POINTS)
r_bc_L, t_bc_L = random_BC_points_L(R,T,n=TRAIN_BC_POINTS)
r_bc_R, t_bc_R = random_BC_points_R(R,T,n=TRAIN_BC_POINTS)
r_ic, t_ic     = random_IC_points(R,n=TRAIN_IC_POINTS)
# print("Sizes: r={}, t={}, r_bc_L={}, t_bc_L={}, r_bc_R={}, t_bc_R={}, r_ic={}, t_ic={}".format(
#     r.size(), t.size(), r_bc_L.size(), t_bc_L.size(), r_bc_R.size(), t_bc_R.size(), r_ic.size(), t_ic.size()))
plt.plot(r.detach().cpu().numpy(),t.detach().cpu().numpy(),'o',ms=1)
plt.plot(r_bc_L.detach().cpu().numpy(),t_bc_L.detach().cpu().numpy(),'o', ms=1)
plt.plot(r_bc_R.detach().cpu().numpy(),t_bc_R.detach().cpu().numpy(),'o', ms=1)
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
        optimizer.zero_grad() # to make the gradients zero
        # RESIDUAL ################################################################ 
        residual = loss_domain(r, t)
        residual_r = torch.autograd.grad(residual, r, create_graph=True, grad_outputs=torch.ones_like(residual))[0]
        # residual_t = torch.autograd.grad(residual, t, create_graph=True, grad_outputs=torch.ones_like(residual))[0]
        loss_dom = torch.mean(residual**2 + residual_r**2)
        # BC ######################################################################
        loss_bc_L = torch.mean(loss_L_BC(r_bc_L,t_bc_L))
        loss_bc_R = torch.mean(loss_R_BC(r_bc_R,t_bc_R))
        # IC ######################################################################
        loss_ic_d = torch.mean(loss_IC(r_ic,t_ic))
        # LOSS ####################################################################
        loss = loss_dom + GAMMA1*loss_bc_L + GAMMA1*loss_bc_R + GAMMA2*loss_ic_d 
        # Calculate and append individual losses to their respective lists
        loss_dom_list.append(loss_dom.item())
        loss_bc_L_list.append(loss_bc_L.item())
        loss_bc_R_list.append(loss_bc_R.item())
        loss_ic_list.append(loss_ic_d.item())

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # 
        # scheduler.step()  # Update learning rate
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch} - Loss: {loss.item():>1.6e} - Learning Rate: {LEARNING_RATE}') # - Learning Rate: {scheduler.get_last_lr()[0]:>1.6e}
    # Adative sample step 
    r_,t_            = random_domain_points(R,T,n=VALID_DOM_POINTS)

    r_bc_L_,t_bc_L_  = random_BC_points_L(R,T,n=VALID_BC_POINTS)
    r_bc_R_,t_bc_R_  = random_BC_points_R(R,T,n=VALID_BC_POINTS)

    r_ic_, t_ic_     = random_IC_points(R,n=VALID_IC_POINTS)
    #
    residual_ = loss_domain(r_, t_)
    residual_r_ = torch.autograd.grad(residual_, r_, create_graph=True, grad_outputs=torch.ones_like(residual_))[0]
    #
    loss_dom_aux  = residual_**2 + residual_r_**2
    loss_bc_L_aux = loss_L_BC(r_bc_L_,t_bc_L_)
    loss_bc_R_aux = loss_R_BC(r_bc_R_,t_bc_R_)
    loss_ic_aux   = loss_IC(r_ic_,t_ic_)   
    #
    idx_dom  = torch.where(loss_dom_aux >= loss_dom_aux.sort(0)[0][-DOM_NEW_POINTS])[0]
    idx_bc_L = torch.where(loss_bc_L_aux >= loss_bc_L_aux.sort(0)[0][-BC_NEW_POINTS])[0]
    idx_bc_R = torch.where(loss_bc_R_aux >= loss_bc_R_aux.sort(0)[0][-BC_NEW_POINTS])[0]
    idx_ic   = torch.where(loss_ic_aux >= loss_ic_aux.sort(0)[0][-IC_NEW_POINTS])[0]
    #
    r_aux = r_[idx_dom].view(-1, 1) # if len(idx_dom) > 0 else torch.tensor([]).view(0, 1)
    t_aux = t_[idx_dom].view(-1, 1) # if len(idx_dom) > 0 else torch.tensor([]).view(0, 1)
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
plt.semilogy(loss_dom_list, label='Domain Loss')
plt.semilogy(loss_bc_L_list, label='Left BC Loss')
# plt.semilogy(loss_bc_R_list, label='Right BC Loss')
# plt.semilogy(loss_ic_list, label='Initial Condition Loss')
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

# CUDA_VISIBLE_DEVICES=0 python spherical_coordinates.py > log_spherical_coordinates_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python spherical_coordinates.py > log_spherical_coordinates_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &