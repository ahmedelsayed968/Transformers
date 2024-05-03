from torch.nn import Module
from torch import nn,Tensor
import torch
from math import sqrt
class CauselCrossAttention(Module):
    def __init__(self, 
                    hidden_size: int,
                    num_heads: int,
                    context_size:int,
                    attn_drop: float = 0.1,
                    out_drop: float = 0.1,
                    bias: bool = True,) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0 , "number of heads must divide hidden_size"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_size = context_size
        self.q_xweight = nn.Linear(in_features=self.hidden_size,
                                   out_features=self.hidden_size,
                                   bias=bias)
        self.kv_yweight = nn.Linear(in_features=self.hidden_size,
                                    out_features=self.hidden_size*2,
                                    bias=bias)
        
        self.project_layer = nn.Linear(in_features=self.hidden_size,
                                       out_features=self.hidden_size)
        self.out_drop = nn.Dropout(p=out_drop)
        self.atten_drop = nn.Dropout(p=attn_drop)
        self.register_buffer(
            name="casual_mask",
            tensor=torch.triu(
                input=torch.ones(size=(context_size,context_size),dtype=torch.bool).\
                view(1,1,context_size,context_size)
                ,
                diagonal=1
            )
        )
    def forward(self,x:Tensor,y:Tensor,mask:Tensor):
        batch_size,in_seq,in_dim =x.shape
        query  = self.q_xweight(x).\
                        reshape(batch_size,in_seq,self.num_heads,in_dim//self.num_heads).\
                        transpose(1,2)
        
        key,value = self.kv_yweight(y).\
                        reshape(batch_size,in_seq,2,self.num_heads,in_dim//self.num_heads).\
                        transpose(1,3).\
                        unbind(2)
        query_dot_key = query @ key.transpose(-1,-2)
        scale_factor = key.size(-1)
        atten = query_dot_key/sqrt(scale_factor)
        all_masks = self.casual_mask[:,:,:in_seq,:in_seq] + mask.view(batch_size,1,1,in_seq)
        atten = atten.masked_fill(all_masks,float('-inf'))
        scores = atten.softmax(dim=-1)
        scores = self.atten_drop(scores)
        x = scores @ value
        x = x.transpose(1,2)
        x = x.reshape(batch_size,in_seq,in_dim)
        return self.out_drop(self.project_layer(x))
    