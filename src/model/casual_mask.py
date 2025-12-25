import torch
def casual_mask(seq_length,device='cpu'):
    mask=torch.trill(torch.ones(seq_length,seq_length)).unsqueeze(0).unsqueeze(0)
    return mask.bool()
    