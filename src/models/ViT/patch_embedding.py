import torch
from torch.nn import Module,Conv2d

class PatchEmbedding(Module):
    def __init__(self,d_model:int,patch_size:int,in_channels:int) -> None:
        super(PatchEmbedding,self).__init__()
        self.conv = Conv2d(in_channels=in_channels,
                           out_channels=d_model,
                           kernel_size=patch_size,
                           stride=patch_size,
                           )
    
    def forward(self,image):
        x:torch.Tensor = self.conv(image)
        batch_size,channels,height,width = x.shape

        x = x.permute(2,3,0,1) 
        x = x.view(height*width,batch_size,channels) # column vector of each patch , d_model
        return x

    

