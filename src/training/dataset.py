import torch
from torch.utils.data import Dataset

class Train_dataset(Dataset):
    def __init__(self,path,block_size):
        self.tokens=torch.load(path)
        self.block_size=block_size
        
    def __len__(self):
        return len(self.tokens)-self.block_size
    
    def __getitem__(self, index):
        x=self.tokens[index:index+self.block_size]
        y=self.tokens[index+1:index+self.block_size+1]
        
        return {'input_ids':x,
                'labels':y}