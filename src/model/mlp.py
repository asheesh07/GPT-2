import torch.nn as nn
import torch.nn.functional as F
class FeedForward(nn.Module):
    def __init__(self,embed_dim,expansion_factor):
        super().__init__()
        hidden_dim=expansion_factor*embed_dim
        
        self.fc1=nn.Linear(embed_dim,hidden_dim)
        self.act=F.GELU()
        self.fc2=nn.Linear(hidden_dim,embed_dim)
        
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        
        return x
        