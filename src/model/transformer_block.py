import torch
import torch.nn as nn

from .attention import MultiHeadAttention,BlockAttention,LinearAttention,GatedAttention

class TransformerBlock(nn.Module):
  def __init__(self,embed_size,heads,dropout,forward_expansion,attention_type='multi'):
    super(TransformerBlock,self).__init__()
    if attention_type=='multi':
      self.attention=MultiHeadAttention(embed_size,heads)
    elif attention_type=='block':
      self.attention=BlockAttention(embed_size,block_size=16)
    elif attention_type=='linear':
      self.attention=LinearAttention(embed_size,heads)
    elif attention_type=='gated':
      self.attention=GatedAttention(embed_size,heads)
    else:
      raise ValueError("Invalid attention type")
    
    self.norm1=nn.LayerNorm(embed_size)
    self.norm2=nn.LayerNorm(embed_size)
    self.feed_forward=nn.Sequential(
        nn.Linear(embed_size,forward_expansion*embed_size),
        nn.GELU(),
        nn.Linear(forward_expansion*embed_size,embed_size)
    )
    self.dropout=nn.Dropout(dropout)

  def forward(self,x,mask=None):
    attention=self.attention(self.norm1(x),mask)
    x=x+self.dropout(attention)

    forward=self.feed_forward(self.norm2(x))
    x=x+self.dropout(forward)

    return x

