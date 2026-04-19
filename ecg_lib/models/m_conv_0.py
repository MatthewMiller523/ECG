#model template

import torch
import torch.nn as nn

#creates a model
class conv_class_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        self.conv2s = nn.Conv2d(1, 16, kernel_size=3, padding =1)
        self.ac1 = nn.Softsign()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16 * 1000 * 12, 64)   # *2 for BiLSTM
        self.ac2 = nn.Softsign()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self,x):
        out, _ = self.lstm(x)
        
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        
        return out
        
        