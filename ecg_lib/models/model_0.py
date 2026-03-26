import torch
import torch.nn as nn

class Classifier_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)   # *2 for BiLSTM
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self,x):
        out, _ = self.lstm(x)
        
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        
        return out
        
        