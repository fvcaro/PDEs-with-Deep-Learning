"""
@author: Felipe V. Caro
"""

import torch
from torch import nn

x = torch.tensor([1., 2.], requires_grad = True).view(-1,1)
print('x: ',x)

class PINN(nn.Module):
    def __init__(self) -> None:
        super(PINN,self).__init__()
        self.layer01 = nn.Linear(1,1)
    
    def forward(self,x):
        inputs = x
        output = torch.sigmoid(self.layer01(inputs))
        return output
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pinn = PINN().to(device)
print(pinn.state_dict())

u = pinn(x)

u_x  = torch.autograd.grad(outputs=u, 
                           inputs=x,
                           create_graph=True,
                           grad_outputs=torch.ones_like(u)
                           )[0]
print(u_x)

# gradient check!
a = pinn.state_dict()['layer01.weight']
b = pinn.state_dict()['layer01.bias']
print(torch.sigmoid(a*x + b)*(1 - torch.sigmoid(a*x + b))*a)

u_xx = torch.autograd.grad(outputs=u_x, 
                           inputs=x,
                           grad_outputs=torch.ones_like(u_x)
                           )[0]
print(u_xx)

# gradient check!
p1 = torch.sigmoid(a*x + b)*(1 - torch.sigmoid(a*x + b))**2*a**2
p2 = - torch.sigmoid(a*x + b)**2*(1 - torch.sigmoid(a*x + b))*a**2
print(p1 + p2)