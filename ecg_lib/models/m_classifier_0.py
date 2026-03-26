#example model
#BiLSTM, FC1, tanh, FC2

import torch
import torch.nn as nn

#creates the model
#I may put a bunch of classifiers in one file. If so, I'll move and rename as necessary
class Classifier_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        #create the lstm layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, #make it a Bidirectional LSTM layer
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)   # *2 for BiLSTM
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(64, num_classes)       #put loss function in calling script
        
    def forward(self,x):
        out, _ = self.lstm(x)
        
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        
        return out
        
        