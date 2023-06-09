import torch

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device('cuda')
else:
    print("GPU is not available, using CPU")
    device = torch.device('cpu')

print("Using device:", device)
