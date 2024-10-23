import torch

# Check if GPU is available, but force using CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Sample tensor on CPU
x = torch.randn(10, 10).to(device)
print(f"Tensor is on {x.device}")
