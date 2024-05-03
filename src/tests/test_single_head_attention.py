import torch
import pytest

from ..models.attention import SingleHeadAttention
from pytest_cases import param_fixture
hidden_size = param_fixture("hidden_size",[128, 256, 512])

@pytest.fixture
def attention_module(hidden_size):
    return SingleHeadAttention(hidden_size=hidden_size)

@pytest.mark.parametrize("hidden_size", [128, 256, 512])
def test_forward(attention_module, hidden_size):
    batch_size = 32
    input_seq = 10
    input_dim = hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    output = attention_module(x)
    # model.hidden_size = hidden_size
    # output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, input_seq, hidden_size)
    
    # Check if the output is not None
    assert output is not None
    
    # Check if gradients can be computed
    output.sum().backward()
    assert x.grad is None
