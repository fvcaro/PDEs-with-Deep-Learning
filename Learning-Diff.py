import torch

x1 = torch.tensor([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = torch.tensor([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
y = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.float32)

# Verify the shapes of the tensors
print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)
print("y shape:", y.shape)
