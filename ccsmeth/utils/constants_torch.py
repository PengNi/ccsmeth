import torch

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()


# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
def FloatTensor(tensor, device=0):
    if use_cuda:
        return torch.FloatTensor(tensor).cuda(device)
    return torch.FloatTensor(tensor)


# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
def LongTensor(tensor, device=0):
    if use_cuda:
        return torch.LongTensor(tensor).cuda(device)
    return torch.LongTensor(tensor)
