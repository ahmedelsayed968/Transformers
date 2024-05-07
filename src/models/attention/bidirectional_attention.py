from torch import nn,Tensor,BoolTensor
# from torch.nn import Module
from .multihead_attention import MultiHeadAttention
from math import sqrt
class BidirectionalAttention(MultiHeadAttention):
    def __init__(self,
                    hidden_size:int,
                    num_heads:int,
                    attn_drop: float = 0.1,
                    out_drop: float = 0.1,
                    bias:bool=True ) -> None:
        super().__init__(hidden_size,num_heads,bias)

        self.attn_drop = nn.Dropout(p = attn_drop) # apply after the softmax
        self.out_drop = nn.Dropout(p=out_drop) # apply to the output layer(project)
    
    def forward(self,x:Tensor,mask: BoolTensor):
        assert isinstance(x,Tensor) ,"input must be of type tensor!"

        batch_size,in_seq,in_dim = x.shape
        # calculate the q,v,k from the input
        x = self.qvk_weight(x) # Batch,in_seq, hidden_size*3
        # Reshape for projecting purposes
        x = x.reshape(batch_size,in_seq,3,self.num_heads,in_dim//self.num_heads)
        # transpose to make 
        x = x.transpose(1,3) # batch, Num_Heads,3,in_seq,in_dim//self.num_heads
        query,value,key = x.unbind(2) # batch, Num_Heads,in_seq,in_dim//self.num_heads
        query_dot_key = query@key.transpose(-2,-1) # batch, Num_Heads,in_seq,in_seq
        scaled_factor = key.size(-1) # d_k
        atten = query_dot_key/sqrt(scaled_factor)
        # before get the softmax
        # we have to remove the effect of pad_tokens
        # so using Mask tensor for this purpose
        # by give the -inf to the pad_token that will corresonde to 0 score softmax
        atten = atten.masked_fill(mask.view(batch_size,1,1,in_seq),float('-inf'))
        
        scores = atten.softmax(dim=-1)
        scores = self.attn_drop(scores)
        self_atten = scores@value # batch, Num_Heads,in_seq,in_dim//self.num_heads
        # concatenate the heads by transpose and reshaping
        x = self_atten.transpose(2,1) # batch,in_seq,Num_Heads,in_dim//self.num_heads
        x = x.reshape(batch_size,in_seq,in_dim)
        # pass the input the projected layer
        q = query.transpose(2,1).reshape(batch_size,in_seq,in_dim)
        return q,self.out_drop(self.project_layer(x))
    



