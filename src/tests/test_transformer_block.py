from ..models import TransformerBlock
from pytest_cases import parametrize,fixture
from .test_bidirectional_attention import get_model
# from .test_casual_attention import get_model as get_atten2
# from .test_cross_attention import get_model 
from ..models.transformer import NormType
from ..models import FeedForwardNetwork
from torch.nn import GELU,SELU
import numpy as np
import torch
@fixture
@parametrize('attention_module',[get_model])
@parametrize('norm_type',[NormType.POST,NormType.PRE])
@parametrize("expand_size",[128,264,512,1024])
@parametrize("drop_rate",list(np.linspace(0.1,0.5,num=5)))
@parametrize("bias",[True,False])
@parametrize("activation",[GELU,SELU])
def get_transformer_block(attention_module,
                          expand_size,
                          drop_rate,
                          bias,
                          activation,
                          norm_type):
    if attention_module.hidden_size > expand_size:
        return None
    ffd_module = FeedForwardNetwork(
        attention_module.hidden_size,
        expand_size,
        drop_rate,
        bias,
        activation
    )
    return TransformerBlock(attention_module,
                          ffd_module,
                          norm_type)

def test_forward(get_transformer_block):
    model = get_transformer_block
    if not model:
        return True
    batch_size = 32
    input_seq = 10
    input_dim = model.attention_block.hidden_size
    x = torch.randn(batch_size, input_seq, input_dim)
    y = torch.randn(batch_size, input_seq, input_dim)
    mask = torch.zeros(size=(batch_size,input_seq), dtype=torch.bool)
    output = model(x,mask,y)
    # Check output shape
    assert output.shape == (batch_size, input_seq, input_dim)    
