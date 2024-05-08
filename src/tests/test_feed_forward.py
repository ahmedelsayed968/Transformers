from ..models.mlp import FeedForwardNetwork
from pytest_cases import fixture,parametrize
from torch.nn import GELU,SELU,SiLU,ReLU
import numpy as np
import torch
@fixture
@parametrize("hidden_size",[128,264,512,1024])
@parametrize("expand_size",[128,264,512,1024])
@parametrize("drop_rate",list(np.linspace(0.1,0.5,num=5)))
@parametrize("bias",[True,False])
@parametrize("activation",[GELU,SELU])
def get_ffd( hidden_size,
                expand_size,
                drop_rate,
                bias,
                activation):
    if hidden_size > expand_size:
        return None
    return FeedForwardNetwork(  hidden_size,
                expand_size,
                drop_rate,
                bias,
                activation)
@parametrize('batch_size',[64])
@parametrize('input_seq',[10])
def test_forward(get_ffd,batch_size,input_seq):
    model = get_ffd
    if not model:
        assert True

    input_dim = model.hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    output = model(x)
    # Check output shape
    assert output.shape == (batch_size, input_seq, input_dim)
    
    # Check if the output is not None
    assert output is not None 