from ..attention.attention import Attention
from ..attention import BidirectionalAttention,CausalAttention,CauselCrossAttention
from ..mlp.feedforward import FeedForwardNetwork
from torch.nn import Module,LayerNorm
from enum import Enum
from torch import Tensor

class NormType(Enum):
    POST = 0
    PRE = 1

class TransformerBlock(Module):
    def __init__(self,
                 attention_module:Attention,
                 ffd_module:FeedForwardNetwork,
                 norm_type:NormType,
                 ) -> None:
        super().__init__()
        assert isinstance(attention_module,(BidirectionalAttention,CauselCrossAttention,CausalAttention))

        self.attention_block = attention_module
        self.ffd_block  = ffd_module
        self.norm_type  = norm_type
        self.norm1 = LayerNorm(self.attention_block.hidden_size)
        self.norm2 = LayerNorm(self.attention_block.hidden_size)
        

    def forward(self,
                x:Tensor,
                mask:Tensor,
                y:Tensor=None)->Tensor:
        if self.norm_type == NormType.POST:
            return self._post_norm_forward(x,mask,y)
        elif self.norm_type == NormType.PRE:
            return self._pre_norm_forward(x,mask,y)
    
    def _post_norm_forward(self,
                           x:Tensor,
                           mask:Tensor,
                           y:Tensor)->Tensor:
        
        if not isinstance(self.attention_block,CauselCrossAttention):
            x = self.norm1(x + self.attention_block(x,mask)[1])
            x = self.norm2(self.ffd_block(x)+x)
            return x
        else:
            assert y is not None
            x = self.norm1(x + self.attention_block(x,y,mask)[1])
            x = self.norm2(self.ffd_block(x)+x)
            return x
    def _pre_norm_forward(self,
                          x:Tensor,
                          mask:Tensor,
                          y:Tensor)->Tensor:
        if not isinstance(self.attention_block,CauselCrossAttention):
            x  = x  + self.attention_block(self.norm1(x),mask)[1]
            x  = x + self.ffd_block(self.norm2(x))
            return x 
        else:
            assert y is not None
            x  = x  + self.attention_block(self.norm1(x),y,mask)[1]
            x  = x + self.ffd_block(self.norm2(x))
            return x 