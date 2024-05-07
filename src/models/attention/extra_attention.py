import torch
from torch.nn import Linear,Module
from .attention import Attention
from torch import Tensor
from .cross_attention import CauselCrossAttention
from .bidirectional_attention import BidirectionalAttention
from .causal_attention import CausalAttention
from .single_attention import SingleHeadAttention
class AttentionOnAttention(Module):
    def __init__(self,
                    attention_module:Attention,
                ):
        super().__init__()
        if isinstance(attention_module,SingleHeadAttention):
            raise NotImplemented("Not Supported for Single Head attention!")
        self.attention_module = attention_module 
        self.linear_1 = Linear(in_features=self.attention_module.hidden_size*2,
                               out_features=self.attention_module.hidden_size)
        
        self.linear_2 = Linear(in_features=self.attention_module.hidden_size*2,
                               out_features=self.attention_module.hidden_size)


    def forward(self,
                    x:Tensor,
                    mask:Tensor=None,
                    y:Tensor=None):

        if y is not None and isinstance(self.attention_module,CauselCrossAttention):
            query,attended_vector = self.attention_module(x,y,mask) # batch,in_seq,in_dim
        elif mask is not None and isinstance(self.attention_module,(BidirectionalAttention,CausalAttention)):
            query,attended_vector = self.attention_module(x,mask) # batch,in_seq,in_dim
        else:
            query,attended_vector = self.attention_module(x) # batch,in_seq,in_dim

        # concate step
        info = torch.cat((query,attended_vector),dim=2) # batch,in_seq,in_dim*2
        # pass through seperate linear layers
        information_vector = self.linear_1(info)
        attention_gate     = self.linear_2(info).sigmoid()
        # final stage of attention by multiplication
        return attention_gate * information_vector
    