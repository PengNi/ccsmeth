import torch

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()


# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
def FloatTensor(tensor, device=0):
    if use_cuda:
        return torch.tensor(tensor, dtype=torch.float, device='cuda:{}'.format(device))
    return torch.tensor(tensor, dtype=torch.float)


def FloatTensor_cpu(tensor):
    return torch.tensor(tensor, dtype=torch.float)


# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
def LongTensor(tensor, device=0):
    if use_cuda:
        return torch.tensor(tensor, dtype=torch.long, device='cuda:{}'.format(device))
    return torch.tensor(tensor, dtype=torch.long)
