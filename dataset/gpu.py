import torch

# Check if GPU is available
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

cuda_available = torch.cuda.is_available()
cudnn_available = torch.backends.cudnn.enabled

print(f"CUDA available: {cuda_available}")  # Should return True if CUDA is properly installed
print(f"cuDNN enabled: {cudnn_available}")  # Should return True if cuDNN is properly installed and enabled

