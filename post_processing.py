import torch
import torch.nn as nn
import numpy as np
import os
from matplotlib import pyplot as plt

# Ensure torch is loaded
print('torch version:', torch.__version__)
assert torch.__version__ is not None, 'Torch not loaded properly'

# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f'Number of available GPUs: {num_gpus}')
for i in range(num_gpus):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

# Path to the saved model
model_dir = 'wave_eq_kessence_bigger_perturbation_Tr_Lr'
saved_model_path = os.path.join(model_dir, 'final_trained_model.pth')

# Define a directory to save the Sols
Sols = os.path.join(model_dir, 'Sols')
os.makedirs(Sols, exist_ok=True)

# Define a directory to save the Sols
Figs = os.path.join(model_dir, 'Figs')
os.makedirs(Figs, exist_ok=True)

R = 40
T = 25
dt = 0.02
dx = 0.05

# Define the model class
class Model(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(), seed=42):
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
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)  

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

# Set the default tensor type to DoubleTensor for torch.float32
torch.set_default_dtype(torch.float32)
# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Instantiate the model and move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layer_sizes = [2, 256, 256, 256, 256, 256, 1]  # 5 hidden layers with 256 neurons each
activation = nn.Tanh()
model = Model(layer_sizes, activation).to(device, dtype=torch.float32)
# Use DataParallel with specified GPUs if multiple GPUs are available
use_data_parallel = torch.cuda.device_count() > 1
if use_data_parallel:
    model = nn.DataParallel(model, device_ids=[0, 1])

# Load the saved state dictionary
state_dict = torch.load(saved_model_path, map_location=device)
if use_data_parallel:
    # If DataParallel was used during training, adjust the state_dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = 'module.' + key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)

# Adjust space discretization
x = torch.linspace(0, R, int(R/dx)+1, requires_grad=True).view(-1, 1).to(device)

# Precompute the time steps
time_steps = torch.linspace(0, T, int(T/dt)+1).to(device)

# Loop over time steps
for idx, t_i in enumerate(time_steps):
    # Create a tensor of the same size as x filled with t_i
    t = torch.full_like(x, t_i.item(), requires_grad=True).to(device)

    # Model and Derivatives
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]

    # Move results to CPU and detach
    x_ = x.cpu().detach().numpy()
    u_ = u.cpu().detach().numpy()
    u_x_ = u_x.cpu().detach().numpy()
    u_t_ = u_t.cpu().detach().numpy()  

    filename = os.path.join(Sols, 'pinn_output_t_{}.txt'.format(idx))
    np.savetxt(filename, np.concatenate((x_, u_, u_x_, u_t_), axis=1), fmt='%.18e')

# Load the saved losses
loss_file = os.path.join(model_dir, 'training_losses.npz')
if os.path.exists(loss_file):
    loss_data = np.load(loss_file)
    print(f"Loss file: '{loss_file}' loaded properly.")
    loss_dom_list = loss_data['loss_dom_list']
    loss_bc_R_list = loss_data['loss_bc_R_list']
    loss_bc_L_list = loss_data['loss_bc_L_list']
    loss_ic_list = loss_data['loss_ic_list']
else:
    raise FileNotFoundError(f"Loss file: '{loss_file}' not found. Ensure it exists and the path is correct.")

# Loss Plot
loss_y_min = 1e-11  # Adjust as necessary based on the expected minimum loss value
loss_y_max = 1e1  # Adjust as necessary based on the expected maximum loss value
fig = plt.figure(figsize=(7,6))
# Fontsize of everything inside the plot
plt.rcParams.update({'font.size': 16})
plt.semilogy(loss_ic_list, label='IC Loss')
plt.semilogy(loss_bc_R_list, label='Right BC Loss')
plt.semilogy(loss_bc_L_list, label='Left BC Loss')
plt.semilogy(loss_dom_list, label='Domain Loss')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.ylim(loss_y_min, loss_y_max)  # Set fixed y-axis limits for loss plot
#
filename = os.path.join(Figs, f'training_losses.png')
plt.savefig(filename, dpi=300, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()

# Initialize x tensor for high-resolution plotting
x_high_res = torch.linspace(0, R, 1024, dtype=torch.float32, device=device).view(-1, 1)

# Precompute high-resolution time steps
time_steps_high_res = torch.linspace(0, T, int(2 * T) + 1, dtype=torch.float32, device=device)

# Loop over time steps for high-resolution plots
for t_i in time_steps_high_res:
    t = t_i.expand_as(x_high_res)

    # Model inference
    nn_sol = model(x_high_res, t).cpu().detach().numpy()

    # Plotting
    plt.figure()
    plt.plot(x_high_res.cpu(), nn_sol, linewidth=2)
    plt.title(f't = {t_i.item():.2f}')
    plt.xlim(0, R)
    plt.grid(True)
        
    # Save plot
    filename = os.path.join(Figs, f'pinns_sol_{t_i.item():.2f}.png')
    print(f'Saving plot for t_i = {t_i.item()} at {filename}')  # Debugging statement
    plt.savefig(filename, dpi=300, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
    plt.close()

# CUDA_VISIBLE_DEVICES=3,4 python post_processing.py