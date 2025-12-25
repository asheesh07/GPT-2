import torch
import torch.nn as nn
import torch.nn.functional as F

#For Linear and Gated Linear Attention
def feature_map(x):
  return F.elu(x)+1

class MultiHeadAttention(nn.Module):
  def __init__(self,embed_size,heads):
    super(MultiHeadAttention,self).__init__()
    self.embed_size=embed_size
    self.heads=heads
    
    assert (embed_size % heads ==0 ),"Embed size needs to be divisible by heads"
    self.head_dim=embed_size//heads

    self.values=nn.Linear(embed_size,embed_size,bias=False)
    self.keys=nn.Linear(embed_size,embed_size,bias=False)
    self.query=nn.Linear(embed_size,embed_size,bias=False)

    self.ff_out=nn.Linear(embed_size,embed_size)

  def forward(self,x,mask):
    
    B,N,D=x.shape
    v=self.values(x)
    k=self.keys(x)
    q=self.query(x)

    v=v.reshape(B,N,self.heads,self.head_dim).permute(0,2,1,3)
    k=k.reshape(B,N,self.heads,self.head_dim).permute(0,2,1,3)
    q=q.reshape(B,N,self.heads,self.head_dim).permute(0,2,1,3)

    score=torch.matmul(q,k.transpose(-2,-1))/(self.head_dim**0.5)

    if mask is not None:
      score=score.masked_fill(mask==0,float('-inf'))

    scores=F.softmax(score,dim=-1)

    output=torch.matmul(scores,v).permute(0,2,1,3).reshape(B,N,self.embed_size)

    output=self.ff_out(output)

    return output
def chunk_blocks(x,block_size):
  B,N,D=x.shape
  num_blocks=N//block_size
  x=x[:,:num_blocks*block_size,:]
  return x.reshape(B,num_blocks,block_size,D)

def casual_Attention(q,k,v):
  scores=torch.matmul(q,k.transpose(-2,-1))
  scores=scores/(q.shape[-1]**0.5)
  scores=F.softmax(scores,dim=-1)
  attn_weights=torch.matmul(scores,v)
  return attn_weights

class BlockAttention(nn.Module):
  def __init__(self,dim,block_size ):
    super().__init__(self)
    self.block_size=block_size
    self.wq=nn.Linear(dim,dim)
    self.wk=nn.Linear(dim,dim)
    self.wv=nn.Linear(dim,dim)

    self.out=nn.Linear(dim,dim)

  def forward(self,x):
    B,N,D=x.shape
    block_size=self.block_size

    q=self.wq(x)
    k=self.wk(x)
    v=self.wv(x)

    q=chunk_blocks(q,block_size)
    k=chunk_blocks(k,block_size)
    v=chunk_blocks(v,block_size)

    bn=q.shape[1]

    for i in range(bn):
      q_i=q[:,i]
      k_i=k[:,i]
      v_i=v[:,i]

      local=casual_Attention(q_i,k_i,v_i)

      if i>0:
        k_prev=k_i[:,i-1]
        v_prev=v_i[:,i-1]
        cross_attn=torch.matmul(q_i,k_prev.transpose(-2,-1))/(q_i.shape[-1]**0.5)
        cross_attn=F.softmax(cross_attn,dim=-1)

        output=torch.matmul(cross_attn,v_prev)
      local=torch.cat((local,output),dim=1)

      out=self.out(local)
    return out

class LinearAttention(nn.Module):
  def __init__(self,dim,head=8):
    super().__init__(self)
    self.head_dim=dim//head
    self.heads=head
    self.wq=nn.Linear(dim,dim)
    self.wk=nn.Linear(dim,dim)
    self.wv=nn.Linear(dim,dim)
    self.out=nn.Linear(dim,dim)

  def forward(self,x):
    B,N,D=x.shape
    q=self.wq(x).view(B,N,self.heads,self.head_dim)
    k=self.wk(x).view(B,N,self.heads,self.head_dim)
    v=self.wv(x).view(B,N,self.heads,self.head_dim)

    q=feature_map(q)
    k=feature_map(k)
    
    kv=torch.einsum("bnhd,bnhe->bhde",k,v)
    z=1/(torch.einsum("bnhd,bhd->bnh",q,k.sum(dim=1))+1e-6)
    scores=torch.einsum("bnhd,bhde,bnh->bnhe",q,kv,z)

    out=self.out(scores).reshape(B,N,D)
    return out

def feature_map(x):
  return F.elu(x)+1

class GatedAttention(nn.Module):
  def __init__(self,dim,heads):
    super().__init__()
    self.heads=heads
    self.head_dim=dim//heads

    self.wg=nn.Linear(dim,dim)
    self.wq=nn.Linear(dim,dim)
    self.wk=nn.Linear(dim,dim)
    self.wv=nn.Linear(dim,dim)
    self.out=nn.Linear(dim,dim)

  def forward(self,x):
    B,N,D=x.shape
    q=self.wq(x).view(B,N,self.heads,self.head_dim)
    k=self.wk(x).view(B,N,self.heads,self.head_dim)
    v=self.wv(x).view(B,N,self.heads,self.head_dim)

    gate=torch.sigmoid(self.wg(x))

    q=feature_map(q)
    k=feature_map(k)

    kv=torch.einsum("bnhd,bnhe->bhde",k,v)
    z=1/(torch.einsum("bnhd,bhd->bnh",q,k.sum(dim=1))+1e-6)
    scores=torch.einsum("bnhd,bhde,bnh->bnhe",q,kv,z)

    out=scores.reshape(B,N,D)
    output=gate*out
    return self.out(output)
