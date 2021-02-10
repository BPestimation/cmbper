import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class ARTDataset(Dataset):
    def __init__(self, data_path):
        split = data_path.split('/')[-1].replace('.p', '')
        df = pickle.load(open(data_path,'rb'))
        self.x_all = torch.tensor(df['PPG'], dtype=torch.float32)
        self.y_all = torch.tensor(df['ABP'], dtype=torch.float32)

        print("Total number of <{}> data points: {}".format(split, len(self.x_all)), flush=True)

    def __len__(self):
        return len(self.x_all)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_all[idx], self.y_all[idx]