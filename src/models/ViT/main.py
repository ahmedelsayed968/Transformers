from vit import VisionTransformer
from mlp import ClassificationHead
from torch.nn import Module
import torch
class MyClassifier(Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.base = VisionTransformer(config)
        self.head = ClassificationHead(config=config)
    def forward(self,image):
        x = self.base(image)
        x = self.head(x)
        return x
CONFIG= {
    "d_model": 124,
    "patch_size":16,
    "in_channels":3,
    "max_patches":257,
    "hidden_size":32,
    "num_classes":5,
    "encoder_num":5,
    "nhead":4
}
image = torch.rand(size=(3,256,256))

model = MyClassifier(config=CONFIG)
out = model(image.unsqueeze(0))
print(out.shape)