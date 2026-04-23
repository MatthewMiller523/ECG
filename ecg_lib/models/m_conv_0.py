#model template

import torch
import torch.nn as nn

#creates a model
class conv_class_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        self.conv2s = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.act1 = nn.Softsign()
        self.flat = nn.Flatten()
        self.fct1 = nn.Linear(16 * 1000 * 12, 64)
        self.act2 = nn.Softsign()
        self.fct2 = nn.Linear(64, num_classes)
        
    def forward(self,x):
        out = self.conv2s(x)
        out = self.act1(out)
        out = self.flat(out)
        out = self.fct1(out)
        out = self.act2(out)
        out = self.fct2(out)
        
        return out
        
        