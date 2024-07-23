import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, a, b, c):
        super(Model, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float))

    def forward(self, x, y):
        output = (self.a * x + self.b * y + self.c) ** 2
        return output

a = 2.0
b = 1.0
c = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

model = Model(a, b, c)
model.to(device)

# Batched input points
x = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float, requires_grad=True).to(device)
y = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float, requires_grad=True).to(device)

def df(output, input, order=1):
    """Compute derivatives of output with respect to inputs."""
    for _ in range(order):
        grads = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        output = grads
    return grads

# Compute first derivatives with respect to x and y
u = model(x, y)
u_x = df(u, x)
u_y = df(u, y)

# Compute second derivatives with respect to x and y
u_xx = df(u_x, x, order=1)
u_yy = df(u_y, y, order=1)

# Compute mixed derivatives
u_xy = df(u_x, y, order=1)  # This computes the derivative of u_x with respect to y
u_yx = df(u_y, x, order=1)  # This computes the derivative of u_y with respect to x

expected_u_xx = 2 * a**2  # Expected second derivative with respect to x
expected_u_xy = 2 * a * b  # Expected second mixed derivative
expected_u_yy = 2 * b**2  # Expected second derivative with respect to y

# Print out the values
print("Expected u_xx:", expected_u_xx)
print("Computed u_xx:", u_xx.cpu().detach().numpy())
print("Expected u_xy:", expected_u_xy)
print("Computed u_xy:", u_xy.cpu().detach().numpy())
print("Expected u_yy:", expected_u_yy)
print("Computed u_yy:", u_yy.cpu().detach().numpy())