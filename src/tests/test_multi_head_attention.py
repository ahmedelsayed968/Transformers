import pytest
import torch
from pytest_cases import fixture,parametrize
from ..models.attention import MultiHeadAttention
# hidden_size = param_fixture('hidden_size',[128, 256, 512])
# num_heads  = param_fixture('num_heads',[2,4,8])



@fixture
# @parametrize(hidden_size=hidden_size, num_heads=num_heads)
@parametrize('hidden_size',[128, 256, 512])
@parametrize('num_heads',[2,4,8])
def get_model(hidden_size,num_heads):
    return MultiHeadAttention(hidden_size=hidden_size,
                              num_heads=num_heads)



def test_forward(get_model):
    model = get_model
    batch_size = 32
    input_seq = 10
    input_dim = model.hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    output = model(x)
    # model.hidden_size = hidden_size
    # output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, input_seq, model.hidden_size)
    
    # Check if the output is not None
    assert output is not None
    
    # Check if gradients can be computed
    output.sum().backward()
    assert x.grad is None
