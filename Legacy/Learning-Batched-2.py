import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, a, b, c):
        super(Model, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float))

    def forward(self, x):
        output = (self.a * x[:, 0] + self.b * x[:, 1] + self.c) ** 2
        return output

a = 2.0
b = 1.0
c = 0.5

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Device:', device)
else:
    device = torch.device("cpu")
    print('CUDA is not available. Using CPU.')

model = Model(a, b, c)
model.to(device)

# Batched input points
x = torch.tensor([[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]], dtype=torch.float, requires_grad=True)
x = x.to(device)

u = model(x)

# Compute gradients
gradients = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

u_xx = []
u_yy = []

u_xx_aux = torch.autograd.grad(gradients[:, 0].sum(), x, create_graph=True)[0]
u_yy_aux = torch.autograd.grad(gradients[:, 1].sum(), x, create_graph=True)[0]
u_xx=u_xx_aux[:,0]
u_yy=u_yy_aux[:,1]

print('Second Derivatives:')
print(u_xx)
print(u_yy)