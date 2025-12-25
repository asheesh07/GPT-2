from torch.utils.data import DataLoader
from .dataset import Train_dataset

def Train_dataloader(path,block_size,batch_size,num_workers=4,shuffle=True):
    dataset=Train_dataset(path,block_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True
    )

        