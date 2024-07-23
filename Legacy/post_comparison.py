import torch
import torch.nn as nn
import numpy as np
import os

R = 40
T = 25

VALID_DOM_POINTS = 65536

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

print('torch version:', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
#
layer_sizes = [2,64,64,64,1]
activation = nn.Tanh()
model_64 = Model(layer_sizes,activation).to(device)
model_64.load_state_dict(torch.load('trained_model_gpu_spherical_64',map_location=torch.device(device)))
model_64.eval()
#
layer_sizes = [2,128,128,128,1]
activation = nn.Tanh()
model_128 = Model(layer_sizes,activation).to(device)
model_128.load_state_dict(torch.load('trained_model_gpu_spherical_128',map_location=torch.device(device)))
model_128.eval()

def random_domain_points(R, T, n=8192):
    r = R * torch.rand(n, 1, device=device, requires_grad=True)
    t = T * torch.rand(n, 1, device=device, requires_grad=True)
    return r, t

r, t = random_domain_points(R,T,n=VALID_DOM_POINTS)

u_64  = model_64(r, t)
u_128 = model_128(r, t)

e_abs = torch.sqrt(torch.mean((u_64 - u_128)**2))
print('e_abs: ', e_abs)

e_rel = torch.sqrt(torch.mean((u_64 - u_128)**2) / torch.mean((u_128)**2))*100
print('e_rel: ', e_rel)

# CUDA_VISIBLE_DEVICES=0 python post_comparison.py