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

    def __init__(self, text: List, text1: List, text2: List, label: List):
        self.text = text
        self.text1 = text1
        self.text2 = text2
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return {"text": self.text[index], "text1": self.text1[index], "text2": self.text2[index], "label": self.label[index]}


def get_dataloader(args):
    """ get train dataloader

    Args:
        args: An argparse class

    Returns:
        dataloader
    
    """
    train_data = pd.read_csv(args.train_data_path)
    test, test1, test2, label = train_data['text'].values.tolist(), train_data['text1'].values.tolist(), \
        train_data['text2'].values.tolist(), train_data['label'].values.tolist()
    train_dataset = AugmentSamples(test, test1, test2, label)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, drop_last = False)
    
    return  train_dataloader
    
    
    
