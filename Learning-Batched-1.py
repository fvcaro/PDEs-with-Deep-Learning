import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, a, b, c):
        super(Model, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float))

    def forward(self, x):
        x, y = x.chunk(2, dim=1)  # Separate input into x and y components tuple unpacking
        output = (self.a * x + self.b * y + self.c) ** 2
        return output

a = 2.0  # Value of a
b = 1.0  # Value of b
c = 0.5  # Value of c

model = Model(a, b, c)

x = torch.tensor([[1.0, 0.1]], dtype=torch.float, requires_grad=True)  # 2D input (x, y)
print('Random Points:')
print(x)

u = model(x)

# Compute first derivative
grads = torch.autograd.grad(u, x, create_graph=True, retain_graph=True)
print(grads)
u_x = grads[0]
print(u_x)
print('first derivative: ')
print(u_x[:, 0])

# Compute second derivatives
u_xx = torch.autograd.grad(u_x[:, 0], x, create_graph=True)[0][:, 0]  # Second derivative with respect to x
print(u_xx)
u_xy = torch.autograd.grad(u_x[:, 1], x, create_graph=True)[0][:, 0]  # Second mixed derivative
u_yy = torch.autograd.grad(u_x[:, 1], x, create_graph=True)[0][:, 1]  # Second derivative with respect to y

expected_u_xx = 2 * a**2  # Expected second derivative with respect to x
expected_u_xy = 2 * a * b  # Expected second mixed derivative
expected_u_yy = 2 * b**2  # Expected second derivative with respect to y

print("u_xx: Computed {:.4f}  Expected {:.4f}".format(u_xx.item(), expected_u_xx))
print("u_xy: Computed {:.4f}  Expected {:.4f}".format(u_xy.item(), expected_u_xy))
print("u_yy: Computed {:.4f}  Expected {:.4f}".format(u_yy.item(), expected_u_yy))