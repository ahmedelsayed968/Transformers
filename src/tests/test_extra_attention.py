from ..models.attention import BidirectionalAttention
from ..models.attention import AttentionOnAttention
from ..models.attention import CauselCrossAttention
from ..models.attention import CausalAttention

import torch
from pytest_cases import fixture,parametrize

# hidden_size = param_fixture('hidden_size',[128, 256, 512])
# num_heads  = param_fixture('num_heads',[2,4,8])


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


@fixture
@parametrize('hidden_size',[128, 256, 512])
@parametrize('num_heads',[2,4,8])
@parametrize("attn_drop",[0.1,0.2,0.3])
@parametrize("out_drop",[0.1,0.2,0.3])
@parametrize("context_size",[512,1024,256])
def get_model2(hidden_size,num_heads,attn_drop,out_drop,context_size):
    return CausalAttention(hidden_size=hidden_size,
                              num_heads=num_heads,
                                context_size=context_size,
                              attn_drop=attn_drop,
                              out_drop=out_drop)


@fixture
@parametrize('hidden_size',[128, 256, 512])
@parametrize('num_heads',[2,4,8])
@parametrize("attn_drop",[0.1,0.2,0.3])
@parametrize("out_drop",[0.1,0.2,0.3])
@parametrize("context_size",[512,1024,256])
def get_model3(hidden_size,num_heads,attn_drop,out_drop,context_size):
    return CauselCrossAttention(hidden_size=hidden_size,
                              num_heads=num_heads,
                                context_size=context_size,
                              attn_drop=attn_drop,
                              out_drop=out_drop)



def test_forward(get_model):
    # MODEL 1
    model = get_model
    batch_size = 32
    input_seq = 10
    input_dim = model.hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    mask = torch.zeros(size=(batch_size,input_seq), dtype=torch.bool)
    attent_on_attent = AttentionOnAttention(model)
    output = attent_on_attent(x,mask)

    # Check output shape
    assert output.shape == (batch_size, input_seq, model.hidden_size)
    
    # Check if the output is not None
    assert output is not None
    
    
def test_forward2(get_model2):
        # MODEL 2
    model = get_model2
    batch_size = 32
    input_seq = 10
    input_dim = model.hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    mask = torch.zeros(size=(batch_size,input_seq), dtype=torch.bool)
    attent_on_attent = AttentionOnAttention(model)
    output = attent_on_attent(x,mask)

    # Check output shape
    assert output.shape == (batch_size, input_seq, model.hidden_size)
    
    # Check if the output is not None
    assert output is not None

def test_forward3(get_model3):
    
    # MODEL 3
    model = get_model3
    batch_size = 32
    input_seq = 10
    input_dim = model.hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    y = torch.randn(batch_size, input_seq, input_dim)
    mask = torch.zeros(size=(batch_size,input_seq), dtype=torch.bool)
    attent_on_attent = AttentionOnAttention(model)
    output = attent_on_attent(x,mask,y)

    # Check output shape
    assert output.shape == (batch_size, input_seq, model.hidden_size)
    
    # Check if the output is not None
    assert output is not None
