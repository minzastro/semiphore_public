import torch

CUDA = torch.device('cpu')

def to_cuda(x):
    return torch.Tensor(x).to(CUDA)
