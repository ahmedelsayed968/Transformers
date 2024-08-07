import torch
from torch.nn import Module,Linear,ReLU

class ClassificationHead(Module):
    def __init__(self,config):
        super().__init__()
        self.linear1 = Linear(in_features=config['d_model'],
                              out_features=config['hidden_size'])
        self.linear2 = Linear(in_features=config['hidden_size'],
                              out_features=config['num_classes'])
        self.act = ReLU()
    def forward(self,x):
        x = self.act(self.linear1(x))
        return self.linear2(x)


