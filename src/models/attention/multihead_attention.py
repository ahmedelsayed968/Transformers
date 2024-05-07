from torch import nn,Tensor
from .attention import Attention
from math import sqrt
class MultiHeadAttention(Attention):
    def __init__(self,
                 hidden_size:int,
                 num_heads:int,
                  bias:bool=True ) -> None:
        super().__init__()
        assert hidden_size%num_heads == 0  # for projection purposes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qvk_weight = nn.Linear(in_features=hidden_size,
                                    out_features=hidden_size*3,
                                    bias=bias)
        self.project_layer = nn.Linear(in_features=hidden_size,
                                       out_features=hidden_size,
                                       bias=bias)
    
    def forward(self,x:Tensor):
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
        scores = atten.softmax(dim=-1)
        self_atten = scores@value # batch, Num_Heads,in_seq,in_dim//self.num_heads
        # concatenate the heads by transpose and reshaping
        x = self_atten.transpose(2,1) # batch,in_seq,Num_Heads,in_dim//self.num_heads
        x = x.reshape(batch_size,in_seq,in_dim)
        # pass the input the projected layer
        q = query.transpose(2,1).reshape(batch_size,in_seq,in_dim)
        return q,self.project_layer(x)
    



