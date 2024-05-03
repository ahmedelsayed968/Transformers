import torch
from torch.nn import Module
from torch import nn,Tensor
from math import sqrt
class SingleHeadAttention(Module):
    def __init__(self,
                 hidden_size:int,
                 bias:bool=True):
        super().__init__()
        # concate the linear weights of Q,K,V
        # into single layer with the same input size of the hidden size
        # and triple the outputs to preserve WQ,WV,WK
        # For Mutlihead case set head size to be four times smaller than the input dimension.

        self.qwv_weights = nn.Linear(in_features=hidden_size,
                                out_features=(hidden_size//4)*3,
                                bias=bias)
        # the attended tokens back to the original shape. 
        self.project_layer = nn.Linear(in_features=hidden_size//4,
                                       out_features=hidden_size)
        
    def forward(self,x:Tensor):
        batch_size,input_seq,input_dim = x.shape
        # calcuate the qwv for the input by passing it through qwv weights
        # output of the operation Batch,Seq,I//4*3
        # reshape them to batch,seq,3,I//4
        # unbind the 3rd dim 
        # unpacking the outputs
        query,value,key  = self.qwv_weights(x).reshape(batch_size,input_seq,3,input_dim//4).unbind(dim=2)
        # k,v,or q is of dim Batch,In_seq,in_dim//4
        # Batch,In_seq,In_seq = Batch,In_seq,in_dim//4  @ Batch,in_dim//4,In_seq
        q_dot_k = query@key.transpose(-2,-1)  # Batch,In_seq,In_seq
        scale_factor = key.size(-1)
        atten = q_dot_k / sqrt(scale_factor)
        # pass the atten to softmax to get the scores
        scores  = atten.softmax(dim=-1)
        # multiple the scores by value matrix 
        # Batch,In_seq,in_dim//4 = Batch,In_seq,In_seq @ Batch,In_seq,in_dim//4
        encoded_x = scores @ value
        # project the encoded_x to have the same hidden_size
        return self.project_layer(encoded_x)
