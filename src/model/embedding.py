import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,vocab_size,embed_size,max_length):
        super().__init__()
        self.word_embedding=nn.Embedding(vocab_size,embed_size)
        self.positional_embedding=nn.Embedding(max_length,embed_size)
    def forward(self,x):
        B,N=x.shape
        tok_emb=self.word_embedding(x)
        pos_emb=self.positional_embedding(torch.arange(0,N,device=x.device).unsqueeze(0))
        
        return tok_emb+pos_emb
    