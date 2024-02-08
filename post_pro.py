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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

model = Model().to(device)
model.load_state_dict(torch.load('trained_model_gpu_spherical',map_location=torch.device(device)))
model.eval()

# Define a directory to save the figures
save_dir = 'Figs'
os.makedirs(save_dir, exist_ok=True)

from matplotlib.animation import FuncAnimation
# plt.style.use('seaborn-pastel')

fig = plt.figure(figsize=(8,6))
ax = plt.axes(xlim=(0, R), ylim=(-2., 2.))
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
    x_tr = torch.linspace(0,R,512).view(-1,1)
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
                     frames=np.linspace(0, 25, 51), 
                     blit=True
                    )
save_filename = os.path.join(save_dir, 'final_wave_animation_gpu_test.gif')
anim.save(save_filename, writer='imagemagick')