import torch
from torch.nn import Module,GELU,Linear,Dropout
from torch import Tensor
class FeedForwardNetwork(Module):
    def __init__(self,
                 hidden_size:int,
                 expand_size:int,
                 drop_rate:float=0.2,
                 bias:bool=True,
                 activation:Module=GELU) -> None:
        super().__init__()
        assert hidden_size <= expand_size , "Must Map the input to same or heigher Space!"
        self.hidden_size = hidden_size
        self.fc1 = Linear(in_features=hidden_size,
                          out_features=expand_size,
                          bias=bias)
        self.fc2  = Linear(in_features=expand_size,
                           out_features=hidden_size,
                           bias=bias)
        self.act = activation()
        self.drop = Dropout(p=drop_rate)


    def forward(self,x:Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
