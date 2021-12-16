import torch
import numpy as np
import argparse
from typing import List
from tqdm.auto import tqdm

from torch.utils import data

from dataloader import get_dataloader
from utils.utils import set_global_random_seed, char2id
from model import sccl



def get_argparser(argv:List):
    """get args

    Args:
        argv: sys.argv[1:]

    Returns:
        args

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type = str, default = 'train/data/ESG_News.csv', help = 'the path of train data')
    parser.add_argument('--batch_size', type='int', default = 2, help ='batch size')
    parser.add_argument('--gpu', type = str, default = 'cpu', help = 'cpu or gpu')
    parser.add_argument('--seed', type = int, default = 10, help = 'the global seed')
    parser.add_argument('--model', type = str ,default = '' ,help = 'the pre_train language model' )
    parser.add_argument('--max_len', type = int, default = 32, help = 'the max lenght of sentence')
    parser.add_argument('--num_classes', type = int, default= 3 ,help = 'the class num of news or text')
    parser.add_argument('--tempreture', type = float, default = 0.3 help = 'the tempreture of  instance_CL_loss')
    parser.add_argument('--alpha', type = 1, default = 1, help = 'the degree of freedom of the Student’s t-distribution')
    parser.add_argument('--max_iter', type = int, default = 10, help = 'the num of max iters')

    args = parser.parse_args()
    return args

def train(args):

    set_global_random_seed(args)
    dataloader = get_dataloader(args)

    sccl_model = sccl(dataloader,args)
    sccl_model.train()
    for i in tqdm(range(args.max_iter)):
        for i,batch in tqdm(enumerate(dataloader)):
            loss = sccl_model.forward(batch)





    return  None




    