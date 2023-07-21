import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, a, b, c):
        super(Model, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))

    def forward(self, x, y):
        output = (self.a * x + self.b * y + self.c) ** 2
        return output

a = 2.0  # Value of a
b = 1.0  # Value of b
c = 0.5  # Value of c

model = Model(a, b, c)

x = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
y = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)

u = model(x, y)

# Compute derivatives
grads = torch.autograd.grad(u, (x, y), create_graph=True)
u_xx = torch.autograd.grad(grads[0], x)[0]

expected_u_xx = 2 * a**2  # Expected second derivative

print("Computed u_xx:", u_xx.item())
print("Expected u_xx:", expected_u_xx)