import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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

# # Define a directory to save the figures
# save_dir = 'Figs'
# os.makedirs(save_dir, exist_ok=True)

# from matplotlib.animation import FuncAnimation
# # plt.style.use('seaborn-pastel')

# fig = plt.figure(figsize=(8,6))
# ax = plt.axes(xlim=(0, R), ylim=(-2., 2.))
# line2, = ax.plot([], [], 
#                  color='tab:blue',
#                  lw=2,
#                  linestyle='--',
#                  label='pinn sol'
#                 )
# ax.text(0.1, 0.9, 
#         "t = ", 
#         bbox={'facecolor': 'white',
#               'alpha': 0.5, 
#               'pad': 5},
#         transform=ax.transAxes, 
#         ha="center")
# #
# ax.legend()
# # #
# def init():
#     #
#     line2.set_data([], [])
#     return line2,
# def animate(i):
#     #####################################################
#     ax.text(0.1, 0.9, 
#             "t= %d" % i,
#             bbox={'facecolor': 'white', 
#                   'alpha': 0.5, 
#                   'pad': 5},
#             transform=ax.transAxes, 
#             ha="center")
# #     #####################################################
# #     x_np = np.linspace(-R,R,512)
#     t = i
# #     y_np = np.exp(-A*((x_np - x0) - c*t)**2)/2 + np.exp(-A*((x_np - x0) + c*t)**2)/2
# #     #####################################################
#     x_tr = torch.linspace(0,R,512).view(-1,1)
#     x_tr = x_tr.to(device)
# #     #
#     t_tr = t*torch.ones_like(x_tr)
#     t_tr = t_tr.to(device)
#     y_tr = model(x_tr,t_tr).cpu().detach().numpy()
# #     #
#     line2.set_data(x_tr.cpu().detach().numpy(), y_tr)
#     return line2,

# anim = FuncAnimation(fig, animate, 
#                      init_func=init,
#                      frames=np.linspace(0, 25, 51), 
#                      blit=True
#                     )
# save_filename = os.path.join(save_dir, 'final_wave_animation_gpu_test.gif')
# anim.save(save_filename, writer='imagemagick')