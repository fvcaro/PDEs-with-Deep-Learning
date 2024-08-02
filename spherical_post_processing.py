import torch
import torch.nn as nn
import numpy as np
import os

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

# Path to the saved model
main_dir = 'Figs_spherical_wave_eq_128_3'
saved_model_path = os.path.join(main_dir, 'final_trained_model.pth')

# Instantiate the model and move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layer_sizes = [2, 128, 128, 128, 128, 1]  # 4 hidden layers with 128 neurons each
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

# Define a directory to save the files
save_dir = os.path.join(main_dir, 'Sols')
os.makedirs(save_dir, exist_ok=True)

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

    filename = os.path.join(save_dir, 'pinn_output_t_{}.txt'.format(idx))
    np.savetxt(filename, np.concatenate((x_, u_, u_x_, u_t_), axis=1), fmt='%.18e')

# CUDA_VISIBLE_DEVICES=0,1 python spherical_post_processing.py