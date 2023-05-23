"""
@author: Felipe V. Caro
"""

import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Available device: ', device)

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.layer01 = nn.Linear(1,10)
        self.layer02 = nn.Linear(10,50)
        self.layer03 = nn.Linear(50,50)
        self.layer04 = nn.Linear(50,50)
        self.layer05 = nn.Linear(50,10)
        self.layer06 = nn.Linear(10,1)
    
    def forward(self,x):
        inputs      = torch.cat([x], axis=1)
        out_layer01 = torch.tanh(self.layer01(inputs))
        out_layer02 = torch.tanh(self.layer02(out_layer01))
        out_layer03 = torch.tanh(self.layer03(out_layer02))
        out_layer04 = torch.tanh(self.layer04(out_layer03))
        out_layer05 = torch.tanh(self.layer05(out_layer04))
        out_layer06 = self.layer06(out_layer05)
        output      = out_layer06
        return output

T = 2
epochs = 1e3
gamma1 = 100.
gamma2 = 100.
    
def random_domain_points(T,n=2048):
    x = torch.rand(n,1,requires_grad=True)
    t = T*torch.rand(n,1,requires_grad=True)
    return x
def random_BC_points(T,n=128):
    x = torch.randint(2,(n,1),dtype=torch.float32,requires_grad=True)
    t = T*torch.rand(n,1,requires_grad=True)
    return x
def random_IC_points(n=32):
    x = torch.rand(n,1,requires_grad=True)
    t = T*torch.zeros(n,1,requires_grad=True)
    return x

model = Model()

learning_rate = 0.005
max_epochs = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
convergence_data = torch.empty((max_epochs), device=device)

loss_list = []
for epoch in range(int(epochs)):
  try:
    optimizer.zero_grad() # to make the gradients zero
    #
    x = random_domain_points(T)
    u = model(x)
    # Derivatives
    u_x = torch.autograd.grad(outputs=u, 
                              inputs=x,
                              create_graph=True,
                              grad_outputs=torch.ones_like(u)
                              )[0]
    u_xx = torch.autograd.grad(outputs=u_x, 
                               inputs=x,
                               create_graph=True,
                               grad_outputs=torch.ones_like(u_x)
                               )[0]
    residual = - u_xx - 0.1*(4.0*torch.pi)*torch.sin(2.0*torch.pi*x)
    loss_dom = torch.mean(torch.pow(residual,2))
    # BC
    x_bc = random_BC_points(T)
    #print(x_bc)
    #u_bc = torch.tensor([[-1.],[1.]])
    u_bc       = model(x_bc)
    loss_bc    = torch.mean(torch.pow(u_bc - 0.,2))
    # LOSS
    loss = loss_dom + gamma1*loss_bc
    loss_list.append(loss.detach().numpy())
    loss.backward() # This is for computing gradients using backward propagation
    optimizer.step() # 

    convergence_data[epoch] = loss

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

  except KeyboardInterrupt:
    break

plt.semilogy(loss_list)
plt.savefig('loss_plot.png')

x = torch.linspace(0,1,126).view(-1,1)

for t_i in np.linspace(0,2,11):
    t = t_i*torch.ones_like(x)
    nn_sol = model(x).detach().numpy()
    #
    plt.figure()
    plt.plot(x,nn_sol,label='nn')
    exact_sol = 0.1*torch.sin(2.0*torch.pi*x)
    plt.plot(x,exact_sol,label='exact sol')
    plt.title(r'$t_i$:'+str(t_i))
    plt.ylim(-0.15,0.15)
    plt.legend()
    plt.savefig('exact_plot.png')
    print('Plots saved successfully.')