import torch
import torch.nn.functional as F

def edge_loss(pred, target):
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
    k = k.view(1,1,3,3).to(pred.device)
    return F.l1_loss(F.conv2d(pred, k, padding=1), F.conv2d(target, k, padding=1))
