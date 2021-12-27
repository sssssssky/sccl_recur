from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List
import os



class AugmentSamples(Dataset):
    """get dataset

    Args:
        text: short text list
        text1: The expanded short text list 1
        text2: The expanded short text list 2
        label: The label corresponding to each text

    Returns:
        dict{"text": text ,"text1": text1, "text2": text2, "label":label }

    """

    def __init__(self, text: List, text1: List, text2: List, label: List, N_time: int):
        self.text = text
        self.text1 = text1
        self.text2 = text2
        self.label = label
        self.length = len(text)
        self.ntime = N_time

    def __len__(self):
        return len(self.text) * self.ntime

    def __getitem__(self, index):
        return {"text": self.text[index % self.length], "text1": self.text1[index % self.length], 
                "text2": self.text2[index % self.length], "label": self.label[index % self.length]}


def get_dataloader(args, ntimes = 1):
    """ get train dataloader

    Args:
        args: An argparse class

    Returns:
        dataloader
    
    """
    train_data = pd.read_csv(args.train_data_path)
    test, test1, test2, label = train_data['text'].values.tolist(), train_data['text1'].values.tolist(), \
        train_data['text2'].values.tolist(), train_data['label'].values.tolist()
    train_dataset = AugmentSamples(test, test1, test2, label, ntimes)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, drop_last = False)
    
    return  train_dataloader
    
    
    
    
