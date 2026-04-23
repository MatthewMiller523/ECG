#this is a class because I'm going to deal with datasets as an opject. Ideally, this keeps things together
#better than functions. idk

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class ECG_Dataset(Dataset):
    def __init__(self, X, Y):
        '''
        X: [N, seq_len, num_features]
        Y: [N]
        '''
        self.X = torch.as_tensor(X, dtype=torch.float32)
        #self.Y = torch.as_tensor(Y, dtype=torch.long)   #fix this when ready
        self.Y = torch.Tensor(Y.to_numpy(dtype=np.float32))   #fix this when ready

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_i = self.X[idx]
        y_i = self.Y[idx]
        return x_i, y_i
        