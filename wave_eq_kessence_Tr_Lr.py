import torch
print('torch version: ', torch.__version__)
assert torch.__version__ is not None, 'Torch not loaded properly'
import torch.nn as nn
import numpy as np
from time import time
import os
from torch.optim.lr_scheduler import ExponentialLR
# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f'Number of available GPUs: {num_gpus}')
for i in range(num_gpus):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
# Set the default tensor type to DoubleTensor for torch.float32
torch.set_default_dtype(torch.float32)
# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)  
# PDE parameters
L  = 40
T  = 30
x0 = 15
A  = 1.
screened = -5.e-6 # middle_perturbation
# 0. initial_perturbation
# -5.e-6 # middle_perturbation
# -1.e-6 # small_perturbation
# -1.e-5 # bigger_perturbation
# Training parameters
GAMMA1 = 1.
GAMMA2 = 10.
# Collocation points
TRAIN_DOM_POINTS = 262144 #131072
TRAIN_BC_POINTS  = 2048   #1024
TRAIN_IC_POINTS  = 2048   #1024
# Set up optimizer and scheduler
LEARNING_RATE = 0.001
DECAY_RATE = 0.9
DECAY_STEPS = 5000
gamma = DECAY_RATE ** (1 / DECAY_STEPS)
EPOCHS = 300000
# Define the model class
class Model(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(),seed=42):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.seed = seed
        torch.manual_seed(seed)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(self.activation)  
        self.init_weights()  

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Glorot (Xavier) initialization
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)  

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
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
    u_tr = torch.autograd.grad(outputs=u_t, inputs=r,
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
    X = (u_r)**2 - (u_t)**2
    K_X  = -1./2. + screened*(3./8.)*X**2
    K_XX = screened*(3./4.)*X
    #
    res1 = 2*r*K_XX*(u_rr)*(u_r**2)/K_X
    res2 = 4*r*K_XX*(u_t)*(u_tr)*(u_r)/K_X
    res3 = 2*r*K_XX*(u_t**2)*(u_tt)/K_X
    res4 = 2*u_r + r*u_rr - r*u_tt 
    residual = res1 - res2 + res3 + res4 
    loss_dom = residual 
    return loss_dom

def lossBCleft(r_bc,t_bc):
    u_bc    = model(r_bc,t_bc)
    # Derivatives
    u_bc_r  = torch.autograd.grad(outputs=u_bc, inputs=r_bc,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_bc)
                              )[0]
    
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
    
    loss_bc = torch.pow(u_bc_t + u_bc_r + u_bc/r_bc -  0.,2)
    return loss_bc

def lossIC(r_ic,t_ic):
    u_ic     = model(r_ic,t_ic)
    # Derivatives
    u_ic_t   = torch.autograd.grad(outputs=u_ic, 
                              inputs=t_ic,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_ic)
                              )[0]
    
    loss_ic  = torch.pow(u_ic - A*torch.exp(-torch.pow((r_ic - x0),2)),2)
    loss_ic += torch.pow(u_ic_t - 0.,2)
    return loss_ic

def random_domain_points(R, T, n=8192):
    r = R*torch.rand(n, 1, dtype=torch.float32, device=device, requires_grad=True)
    t = T*torch.rand(n, 1, dtype=torch.float32, device=device, requires_grad=True)
    return r, t

def random_BC_points_L(R, T, n=512):
    r = 0*torch.ones((n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T*torch.rand(n, 1, dtype=torch.float32, device=device, requires_grad=True)
    return r, t

def random_BC_points_R(R, T, n=512):
    r = R*torch.ones((n, 1), dtype=torch.float32, device=device, requires_grad=True)
    t = T*torch.rand(n, 1, dtype=torch.float32, device=device, requires_grad=True)
    return r, t

def random_IC_points(R, n=128):
    r = R*torch.rand(n, 1, dtype=torch.float32, device=device, requires_grad=True)
    t = torch.zeros(n, 1, dtype=torch.float32, device=device, requires_grad=True)
    return r, t

# Define a directory to save the results
outputs_dir = 'wave_eq_kessence_middle_perturbation_Tr_Lr'
os.makedirs(outputs_dir, exist_ok=True)

# Instantiate the model and move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layer_sizes = [2, 256, 256, 256, 256, 256, 1]  # 5 hidden layers with 256 neurons each
activation = nn.Tanh()
model = Model(layer_sizes, activation).to(device, dtype=torch.float32)

# Use DataParallel with specified GPUs
use_data_parallel = torch.cuda.device_count() > 1
if use_data_parallel:
    model = nn.DataParallel(model, device_ids=[0, 1, 2])

# Path to the saved model
model_dir = 'wave_eq_kessence_initial_perturbation'
saved_model_path = os.path.join(model_dir, 'final_trained_model.pth')
print(f"Model: '{saved_model_path}' loaded properly.")

# Load the saved state dictionary
state_dict = torch.load(saved_model_path, map_location=device)

if use_data_parallel:
    # Adjust state_dict keys if using DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if it exists
        if key.startswith("module."):
            new_key = key.replace("module.", "", 1)
        else:
            new_key = key
        new_state_dict["module." + new_key] = value  # Add the correct prefix

    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)

# Randomly generate training data
r, t           = random_domain_points(L, T, n=TRAIN_DOM_POINTS)
r_bc, t_bc     = random_BC_points_L(L, T, n=TRAIN_BC_POINTS)
r_bc_R, t_bc_R = random_BC_points_R(L, T, n=TRAIN_BC_POINTS)
r_ic, t_ic     = random_IC_points(L, n=TRAIN_IC_POINTS)

# Save initial sampling points
filename = os.path.join(outputs_dir, f'initial_sampling_points')
np.savez(filename,
         r=r.cpu().detach().numpy(),
         t=t.cpu().detach().numpy(),
         r_bc_R=r_bc_R.cpu().detach().numpy(),
         t_bc_R=t_bc_R.cpu().detach().numpy(),
         r_bc=r_bc.cpu().detach().numpy(),
         t_bc=t_bc.cpu().detach().numpy(),
         r_ic=r_ic.cpu().detach().numpy(),
         t_ic=t_ic.cpu().detach().numpy())

# Set up optimizer with different learning rates for different parts of the model
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
# Training loop
start_time = time()
loss_dom_list  = []
loss_bc_L_list = []
loss_bc_R_list = []
loss_ic_list   = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    # Compute losses (assuming these functions are defined)
    residual = lossRes(r, t)
    loss_dom = torch.mean(torch.pow(residual, 2))
    loss_bc_L = torch.mean(lossBCleft(r_bc, t_bc))
    loss_bc_R = torch.mean(lossBCright(r_bc_R, t_bc_R))
    loss_ic = torch.mean(lossIC(r_ic, t_ic))
    # Total loss
    loss = loss_dom + GAMMA1 * (loss_bc_L + loss_bc_R) + GAMMA2 * loss_ic
    # Record losses
    loss_dom_list.append(loss_dom.item())
    loss_bc_L_list.append(loss_bc_L.item())
    loss_bc_R_list.append(loss_bc_R.item())
    loss_ic_list.append(loss_ic.item())
    # Backward pass and optimization
    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()
    # Print the current learning rate and loss every 2000 epochs
    if epoch % 2000 == 0:
        elapsed_time = (time() - start_time) / 60  # Convert elapsed time to minutes
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f'Epoch: {epoch} - Loss: {loss.item():>1.3e} - Learning Rate: {current_lr:>1.3e} - Elapsed Time: {elapsed_time:.2f} [min]')

    # Save checkpoint
    if epoch % 100000 == 0:
        torch.save(model.module.state_dict(), f"{outputs_dir}/checkpoint_{epoch}.pth")

print('Computing time', (time() - start_time) / 60, '[min]')
# Save the model
filename = os.path.join(outputs_dir, 'final_trained_model.pth')
torch.save(model.module.state_dict(), filename)  # Save when using DataParallel
filename = os.path.join(outputs_dir, f'training_losses')
np.savez(filename,
         loss_dom_list=loss_dom_list,
         loss_bc_R_list=loss_bc_R_list,
         loss_bc_L_list=loss_bc_L_list,
         loss_ic_list=loss_ic_list
        )

# CUDA_VISIBLE_DEVICES=0,1,2 python wave_eq_kessence_Tr_Lr.py > log_wave_eq_kessence_Tr_Lr_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &