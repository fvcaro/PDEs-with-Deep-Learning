import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
#
from matplotlib import pyplot as plt
import numpy as np
from time import time
#
import os
#
import seaborn as sns
sns.set_style("whitegrid")

print('torch version:',torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

#device = torch.device("cpu")
#print('CUDA is not available. Using CPU.')

L     = 40
T     = 30
x0    = 0
A     = 1.
c = 1.

def exact_sol_f(x,t,x0=x0,A=A,c=c):
    aux = torch.exp(-A*((x - x0) - c*t)**2)/2 + torch.exp(-A*((x - x0) + c*t)**2)/2
    return aux

gamma1 = 100.
gamma2 = 100.
#
dom_points = 1024
bc_points  = 64
ic_points  = 32

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.layer01 = torch.nn.Linear(2,10)
        self.layer02 = torch.nn.Linear(10,50)
        self.layer03 = torch.nn.Linear(50,50)
        self.layer04 = torch.nn.Linear(50,50)
        self.layer05 = torch.nn.Linear(50,10)
        self.layer06 = torch.nn.Linear(10,1)
    
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

def loss_1(r,t):
    u = model(r,t)
    # Derivatives
    u_t  = torch.autograd.grad(outputs=u, 
                              inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_tt = torch.autograd.grad(outputs=u_t, 
                              inputs=t,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_t)
                              )[0]
    u_r  = torch.autograd.grad(outputs=u, 
                              inputs=r,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_rr = torch.autograd.grad(outputs=u_r, 
                               inputs=r,
                               create_graph=True,
                               grad_outputs=torch.ones_like(u_r)
                               )[0]
    #
    residual = u_tt - (c**2)*u_rr
    loss_dom = torch.pow(residual,2)
    return loss_dom

#def loss_2(r_bc,t_bc):
#    u_bc    = model(r_bc,t_bc)
#    #
#    u_bc_t  = torch.autograd.grad(outputs=u_bc, 
#                              inputs=t_bc,
#                              create_graph=True,
#                              grad_outputs=torch.ones_like(u_bc)
#                              )[0]
#    u_bc_r  = torch.autograd.grad(outputs=u_bc, 
#                              inputs=r_bc,
#                              create_graph=True,
#                              grad_outputs=torch.ones_like(u_bc)
#                              )[0]
#    #
#    loss_bc = torch.mean(torch.pow(r_bc*u_bc_t - u_bc + r_bc*u_bc_r,2))
#    return loss_bc

def loss_3(r_ic,t_ic):
    u_ic     = model(r_ic,t_ic)
    #
    u_ic_t   = torch.autograd.grad(outputs=u_ic, 
                              inputs=t_ic,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u_ic)
                              )[0]
    #
    loss_ic  = torch.pow(u_ic - torch.exp(-A*torch.pow((r_ic),2)),2)
    loss_ic += torch.pow(u_ic_t - 0.,2)
    return loss_ic

def random_domain_points(b,T,n=8192):
    x = (2*b)*(torch.rand(n,1,device=device,requires_grad=True) - 0.5)
    t = T*torch.rand(n,1,device=device,requires_grad=True)
    return x,t
def random_BC_points(b,T,n=512):
    x = (2*b)*(torch.ones((n,1),dtype=torch.float32,device=device,requires_grad=True) - 0.5)
    t = T*torch.rand(n,1,device=device,requires_grad=True)
    return x,t
def random_IC_points(b,n=128):
    x = (2*b)*(torch.rand(n,1,device=device,requires_grad=True) - 0.5)
    t = torch.zeros(n,1,device=device,requires_grad=True)
    return x,t

torch.manual_seed(42)
model = Model().to(device)

optimizer = torch.optim.Adam(model.parameters(),
                              lr=0.0001)
scheduler = StepLR(optimizer, step_size=2000, gamma=1., verbose=False)  # Learning rate scheduler

# Define a directory to save the figures
save_dir = 'Figs'
os.makedirs(save_dir, exist_ok=True)

x = torch.linspace(-L,L,256).view(-1,1)
#
plt.figure()
exact_sol = exact_sol_f(x,0.)
plt.plot(x,exact_sol,label='exact sol',color='tab:orange')
plt.legend()
save_filename = os.path.join(save_dir, 'wave_Exact_sol.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

r,t        = random_domain_points(L,T,n=dom_points)
#r_bc, t_bc = random_BC_points(L,T,n=bc_points)
r_ic, t_ic = random_IC_points(L,n=ic_points)
plt.plot(r.detach().cpu().numpy(),t.detach().cpu().numpy(),'o',ms=1)
#plt.plot(r_bc.detach().numpy(),t_bc.detach().numpy(),'o')
plt.plot(r_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'points.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

# r = r.to(device)
# t = t.to(device)
# #
# r_ic = r_ic.to(device)
# t_ic = t_ic.to(device)

loss_list = []
t0 = time()
# Initial training set
stop_criteria = 1.
while (stop_criteria > 0.0001): #
    for epoch in range(5000): #5000
        # Track epochs
        #if (epoch%(epochs/10)==0):
        #    print('epoch:',epoch)
        optimizer.zero_grad() # to make the gradients zero
        # RESIDUAL ################################################################   
        loss_dom = torch.mean(loss_1(r,t))
        # BC ######################################################################
        #loss_bc  =  loss_2(r_bc,t_bc)
        # IC ######################################################################
        loss_ic  = torch.mean(loss_3(r_ic,t_ic))
        # LOSS ####################################################################
        loss = loss_dom + gamma2*loss_ic
        loss_list.append(loss.cpu().detach().numpy())
        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # 
        scheduler.step()  # Update learning rate
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} - Loss: {loss.item():>7f} - Learning Rate: {scheduler.get_last_lr()[0]:>7f}")
    # Adative sample step    
    r_,t_        = random_domain_points(L,T,n=10*dom_points)
    r_ = r_.to(device)
    t_ = t_.to(device)
    r_ic_, t_ic_ = random_IC_points(L,n=10*ic_points)
    r_ic_ = r_ic_.to(device)
    t_ic_ = t_ic_.to(device)
    #
    loss_dom_aux = loss_1(r_,t_)
    loss_ic_aux  = loss_3(r_ic_,t_ic_)   
    #
    idx_dom = torch.where(loss_dom_aux >= loss_dom_aux.sort(0)[0][-500])[0]
    idx_ic  = torch.where(loss_ic_aux >= loss_ic_aux.sort(0)[0][-50])[0]
    #
    r_aux = r_[idx_dom].view(-1,1)
    #print(r_[idx_dom])
    t_aux = t_[idx_dom].view(-1,1)
    r = torch.cat((r,r_aux),0)
    t = torch.cat((t,t_aux),0)
    #
    r_ic_aux = r_ic_[idx_ic].view(-1,1)
    t_ic_aux = t_ic_[idx_ic].view(-1,1)
    r_ic = torch.cat((r_ic,r_ic_aux),0)
    t_ic = torch.cat((t_ic,t_ic_aux),0)
    #
    stop_criteria = loss_dom_aux.sort(0)[0][-1].cpu().detach().numpy()[0]
    #
print('computing time',(time() - t0)/60,'[min]')  

loss_dom_aux.sort(0)[0][-1].cpu().detach().numpy()[0]

torch.save(model.state_dict(), 'trained_model_gpu')

plt.figure()
plt.plot(r.detach().cpu().numpy(),t.detach().cpu().numpy(),'o',ms=1)
#plt.plot(r_bc.detach().numpy(),t_bc.detach().numpy(),'o')
plt.plot(r_ic.detach().cpu().numpy(), t_ic.detach().cpu().numpy(), 'o', ms=1)
save_filename = os.path.join(save_dir, 'new_points.png')
plt.savefig(save_filename, dpi=600, facecolor=None, edgecolor=None,
            orientation='portrait', format='png',transparent=True, 
            bbox_inches='tight', pad_inches=0.1, metadata=None)
plt.close()  # Close the figure to release resources

#np.savez('adaptive_sampling_points',x=r.detach().numpy(),t=t.detach().numpy())

# Plotting individual losses
plt.figure(figsize=(10, 6))
plt.semilogy(loss_list, label='Total Loss')
#plt.semilogy(loss_domain_list, label='Domain Loss')
#plt.semilogy(loss_bc_list, label='Boundary Condition Loss')
#plt.semilogy(loss_ic_list, label='Initial Condition Loss')
plt.legend()
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.grid(True, which="both", ls="--")
save_filename = os.path.join(save_dir, 'individual_loss.png')
plt.savefig(save_filename, dpi=300)
plt.close()  # Close the figure to release resources

# np.save('loss_adaptive_sampling_gpu',loss_list)

for t_i in np.linspace(0, T, 11):
    t = t_i * torch.ones_like(x)
    # Move x and t to the same device as the model
    t = t.to(device)
    x = x.to(device)
    nn_sol = model(x, t).cpu().detach().numpy()
    #
    plt.figure()
    plt.plot(x.cpu(),nn_sol,label='nn',linewidth='2')
    exact_sol = exact_sol_f(x.cpu().detach(),t_i)
    plt.plot(x.cpu(),exact_sol,color='tab:orange',label='exact sol')
    plt.title(r'$t_i$:'+str(t_i))
    plt.xlim(-L,L)
    plt.legend()

    # Save the figure with a unique name based on the time step
    save_filename = os.path.join(save_dir, f'figure_t_{t_i}.png')
    plt.savefig(save_filename, dpi=300)  # You can adjust the dpi (dots per inch) for higher resolution
    plt.close()  # Close the figure to release resources

from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

fig = plt.figure(figsize=(8,6))
ax = plt.axes(xlim=(-L+6, L-6), ylim=(-.2, 1.))
line1, = ax.plot([], [],
                 color='tab:orange',
                 lw=2,
                 label='exact sol'
                )
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
#
def init():
    #
    line1.set_data([], [])
    line2.set_data([], [])
    return (line1,line2)
def animate(i):
    #####################################################
    ax.text(0.1, 0.9, 
            "t= %d" % i,
            bbox={'facecolor': 'white', 
                  'alpha': 0.5, 
                  'pad': 5},
            transform=ax.transAxes, 
            ha="center")
    #####################################################
    x_np = np.linspace(-L,L,512)
    t = i
    y_np = np.exp(-A*((x_np - x0) - c*t)**2)/2 + np.exp(-A*((x_np - x0) + c*t)**2)/2
    #####################################################
    x_tr = torch.linspace(-L,L,512).view(-1,1)
    x_tr = x_tr.to(device)
    #
    t_tr = t*torch.ones_like(x_tr)
    t_tr = t_tr.to(device)
    y_tr = model(x_tr,t_tr).cpu().detach().numpy()
    #
    line1.set_data(x_np, y_np)
    line2.set_data(x_tr.cpu().detach().numpy(), y_tr)
    return (line1,line2)

anim = FuncAnimation(fig, animate, 
                     init_func=init,
                     frames=np.linspace(0, 30, 60), 
                     blit=True
                    )
save_filename = os.path.join(save_dir, 'final_wave_animation_gpu.gif')
anim.save(save_filename, writer='imagemagick')

# python cartesian_2_wave_eq.py &> log_file.txt