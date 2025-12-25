import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import Embedding
from .transformer_block import TransformerBlock
class GPT2(nn.Module):
    def __init__(self,vocab_size, embed_size, num_layers, heads, forward_expansion,
max_length, dropout, attention_type="multi"):
        super().__init__()
        self.embedding=Embedding(vocab_size,embed_size,max_length)
        
        self.Layers=nn.ModuleList([
            TransformerBlock(embed_size,heads,dropout,forward_expansion,attention_type)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_size)

    # LM head (weight tying with token embeddings)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

    # Tie weights
        self.lm_head.weight = self.embedding.word_embedding.weight


        self.max_length = max_length
        
    def forward(self,x,mask=None):
        B,N=x.shape
        
        x=self.embedding(x)
        
        for layer in self.Layers:
            x=layer(x,mask)
        x= self.ln_f(x)

        # LM Head (tied weights)
        logits = self.lm_head(x) 

        return logits