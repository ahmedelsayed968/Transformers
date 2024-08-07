import torch
from torch.nn import Module
from patch_embedding import PatchEmbedding
from positional_embedding import LearnedPositionalEmbedding
from torch.nn import TransformerEncoderLayer,ModuleList,Parameter,LayerNorm
class VisionTransformer(Module):
    def __init__(self, config) -> None:
        super(VisionTransformer,self).__init__()
        self.patch_embedder = PatchEmbedding(config['d_model'],
                                        config['patch_size'],
                                        config['in_channels'])
        
        self.positional_embedder = LearnedPositionalEmbedding(config['d_model'],
                                                        config['max_patches'])

        self.transformer_encoder_layers = ModuleList([
            TransformerEncoderLayer(d_model=config['d_model'],
                                    nhead=config['nhead'])
            for i in range(config['encoder_num'])
        ])
        self.cls_token_emb = Parameter(torch.randn(1, 1, config['d_model']), requires_grad=True)
        self.layer_norm = LayerNorm(config['d_model'])

    def forward(self,image):
        x = self.patch_embedder(image)
        # change the the cls_token to fit the number of batches came from patch_emb step
        cls_token = self.cls_token_emb.expand(-1,x.shape[1],-1) 
        x = torch.cat([cls_token,x])
        
        x = self.positional_embedder(x)
        for layer in self.transformer_encoder_layers:
            x= layer(x)
        
        cls_hidden = x[0]
        x = self.layer_norm(cls_hidden)
        return x
    
