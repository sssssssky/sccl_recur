from os import truncate
import random
import torch
import numpy as np
import transformers
from sentence_transformers import SentenceTransformer


def set_global_random_seed(args):
    """set global seed

    Args:
        args: global args

    Returns:
        None

    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model_name(args):
    MODEL_CLASS = {
        "distil": 'distilbert-base-nli-stsb-mean-tokens', 
        "robertabase": 'roberta-base-nli-stsb-mean-tokens',
        "robertalarge": 'roberta-large-nli-stsb-mean-tokens',
        "msmarco": 'distilroberta-base-msmarco-v2',
        "xlm": "xlm-r-distilroberta-base-paraphrase-v1",
        "bertlarge": 'bert-large-nli-stsb-mean-tokens',
        "bertbase": 'bert-base-nli-stsb-mean-tokens',
    }
    return MODEL_CLASS[args.model]


def char2id(tokenizer, batch, args):

    text, text1, text2 = batch['text'], batch['text1'], batch['text2']
    label =batch['label'] if args.gpu == 'cpu' else batch['label'].cuda()
  
    all_text = [text, text1, text2]
    feat = []
    for text_i in all_text:
        ids = tokenizer.batch_encode_plus(text_i, max_length = args.max_len, return_tenser = 'pt', \
            padding = 'longest', truncating = True)
        feat.append(ids)
    return ids,label


def id2embeding(model, ids):


    return None
