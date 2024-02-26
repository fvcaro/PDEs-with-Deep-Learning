import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation

R = 40

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

layer_sizes = [2, 128, 128, 128, 1]
activation = nn.Tanh()
model = Model(layer_sizes, activation).to(device)
model.load_state_dict(torch.load('trained_model_gpu_spherical', map_location=device))
model.eval()

# Define a directory to save the figures
save_dir = 'Ani'
os.makedirs(save_dir, exist_ok=True)

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

plt.show()

# CUDA_VISIBLE_DEVICES=1 python post_animation.py