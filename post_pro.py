import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import seaborn as sns

R = 40
T = 25
dt = 0.02
dx = 0.05

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
                nn.init.xavier_uniform_(layer.weight)  
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

layer_sizes = [2,128,128,128,1]
activation = nn.Tanh()
model = Model(layer_sizes,activation).to(device)
model.load_state_dict(torch.load('trained_model_gpu_spherical',map_location=torch.device(device)))
model.eval()

# Define a directory to save the files
save_dir = 'Sols'
os.makedirs(save_dir, exist_ok=True)

# Adjust space discretization
x = torch.linspace(0, R, int(R/dx)+1, requires_grad=True).view(-1, 1)
x = x.to(device)

for idx, t_i in enumerate(np.linspace(0, T, int(T/dt)+1)):
    t = t_i * torch.ones_like(x, requires_grad=True)
    t = t.to(device)
    # Model and Derivatives
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    # Move x and t to the same device as the model
    x_ = x.cpu().detach().numpy()
    u_ = u.cpu().detach().numpy()
    u_x_ = u_x.cpu().detach().numpy()
    u_t_ = u_t.cpu().detach().numpy()  

    filename = os.path.join(save_dir,'pinn_output_t_{}.txt'.format(idx))
    np.savetxt(filename, np.concatenate((x_, u_, u_x_, u_t_), axis=1), fmt='%.18e')

# CUDA_VISIBLE_DEVICES=1 python post_pro.py