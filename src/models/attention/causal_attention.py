from .bidirectional_attention import BidirectionalAttention
import torch
from math import sqrt
class CausalAttention(BidirectionalAttention):
    def __init__(self,
                  hidden_size: int,
                    num_heads: int,
                    context_size:int,
                      attn_drop: float = 0.1,
                        out_drop: float = 0.1,
                          bias: bool = True,
                          ) -> None:
        super().__init__(hidden_size, num_heads, attn_drop, out_drop, bias)
        self.register_buffer(
            name="casual_mask",
            tensor=torch.triu(
                torch.ones(size=(context_size,context_size),dtype=torch.bool).\
                view(1,1,context_size,context_size),
                diagonal=1
            )
        )
    
    def forward(self, x: torch.Tensor, mask: torch.BoolTensor):
        # mask: batch_size,in_seq
        batch_size,in_seq,in_dim = x.shape
        x = self.qvk_weight(x)
        x = x.reshape(batch_size,in_seq,3,self.num_heads,in_dim//self.num_heads)
        x = x.transpose(1,3)
        query,value,key = x.unbind(2)
        query_dot_key = query @ key.transpose(-2,-1)
        scale_factor = key.size(-1)
        atten = query_dot_key/sqrt(scale_factor) # batch_size,num_heads,in_seq,in_seq
        # apply all masks before the step of scoring
        all_masks = self.casual_mask[:,:,:in_seq,:in_seq] + mask.view(batch_size,1,1,in_seq)
        atten = atten.masked_fill(all_masks,float('-inf'))
        scores = atten.softmax(dim=-1)
        # apply atten_drop
        scores = self.attn_drop(scores)
        x = scores @ value  # batch_size,num_heads,in_seq,in_dim//self.num_heads
        # step to combine all heads
        # 1. transpose 
        x = x.transpose(1,2)
        # 2. Reshape to the origina dim of x
        x = x.reshape(batch_size,in_seq,in_dim)
        q = query.transpose(2,1).reshape(batch_size,in_seq,in_dim)
        return q,self.out_drop(self.project_layer(x))