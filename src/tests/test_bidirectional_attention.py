from ..models.attention import BidirectionalAttention
import torch
from pytest_cases import fixture,parametrize


@fixture
# @parametrize(hidden_size=hidden_size, num_heads=num_heads)
@parametrize('hidden_size',[128, 256, 512])
@parametrize('num_heads',[2,4,8])
@parametrize("attn_drop",[0.1,0.2,0.3])
@parametrize("out_drop",[0.1,0.2,0.3])
def get_model(hidden_size,num_heads,attn_drop,out_drop):
    return BidirectionalAttention(hidden_size=hidden_size,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              out_drop=out_drop)




def test_forward(get_model):
    model = get_model
    batch_size = 32
    input_seq = 10
    input_dim = model.hidden_size
    mask = torch.zeros(size=(batch_size,input_seq), dtype=torch.bool)
    x = torch.randn(batch_size, input_seq, input_dim)
    output = model(x,mask)
    # model.hidden_size = hidden_size
    # output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, input_seq, model.hidden_size)
    
    # Check if the output is not None
    assert output is not None
    
    # Check if gradients can be computed
    output.sum().backward()
    assert x.grad is None
