import torch
from torch.nn import Module,Parameter

class LearnedPositionalEmbedding(Module):
    def __init__(self,d_model:int,max_patches:int):
        super(LearnedPositionalEmbedding,self).__init__()
        self.pe = Parameter(torch.zeros(size=(max_patches,1,d_model)),requires_grad=True)

    def forward(self,embed_image:torch.Tensor):
        """
            embed_image: (patches,batch_size,d_model)
        """
        patches = embed_image.shape[0]
        corresponding_pe = self.pe[:patches,:,:]
        return embed_image+corresponding_pe

