from typing import Tuple
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.clustering_utils import get_kmean_center
from utils.utils import get_model_name
from dataloader import get_dataloader
from utils.clustering_utils import get_cluster_prob, get_auxiliary_distribution


class sccl(nn.Module):
    def __init__(self, argsss: argparse.ArgumentParser):
        super(sccl, self).__init__()
        self.args = argsss
        self.embedding_model = SentenceTransformer(get_model_name(self.args))
        self.dataloader = get_dataloader(self.args)
        self.tokenizer = self.embedding_model[0].tokenizer
        self.embedding_model_without_pool = self.embedding_model[0].auto_model
        self.embedding_size = self.embedding_model_without_pool.config.hidden_size

        self.instance_cl = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_size, 128)
        )
        initial_cluster_center = torch.tensor(
            get_kmean_center(self.embedding_model, self.dataloader, self.args)[
                0], dtype=torch.float, requires_grad=True
        )

        self.cluster_center = nn.Parameter(initial_cluster_center)

    def char2id(self, batch: dict) -> Tuple[torch.tensor, np.ndarray]:
        """Convert text to id
        
        Args:
            batch: text for each batch
        
        Returns:
            ids
                
        """
        text, text1, text2 = batch['text'], batch['text1'], batch['text2']
        label = batch['label'] if self.args.gpu == 'cpu' else batch['label'].cuda()

        all_text = [text, text1, text2]
        feat = []
        for text_i in all_text:
            ids = self.tokenizer.batch_encode_plus(text_i, max_length=self.args.max_len, return_tensors='pt',
                                                   padding='longest', truncation=True)
            feat.append(ids)

        feat = [{key: fea[key].cuda() for key in fea} for fea in feat]
        return feat, label.cpu().numpy()

    def id2embedding(self, ids: torch.tensor, pooling='mean') -> torch.tensor:
        """Convert id to embedding
        
        Args:
            ids: id representation of each text
            
        Returns:
            embedding: vector representation of text
        
        """
        out = self.embedding_model_without_pool.forward(**ids)
        attention_mask = ids['attention_mask'].unsqueeze(-1)
        cls_out = out[0]
        embedding = torch.sum(cls_out*attention_mask, dim=1) / \
            torch.sum(attention_mask, dim=1)
        return embedding

    def forward(self, batch: dict) -> dict:
        return_dic = {}
        ids, _ = self.char2id(batch)
        embedding0, embedding1, embedding2 = self.id2embedding(
            ids[0]), self.id2embedding(ids[1]), self.id2embedding(ids[2])

        return_dic['embedding0'] = embedding0
        return_dic['embedding1'] = embedding1
        return_dic['embedding2'] = embedding2
        # instance_CL_loss
        feat1 = F.normalize(self.instance_cl(embedding1), dim=1)
        feat2 = F.normalize(self.instance_cl(embedding2), dim=1)
        return_dic['feat1'] = feat1
        return_dic['feat2'] = feat2

        # clusterring_loss
        _, cluster_prob = get_cluster_prob(
            embedding0, self.cluster_center, self.args)
        _, cluster_prob1 = get_cluster_prob(
            embedding0, self.cluster_center, self.args)
        _, cluster_prob2 = get_cluster_prob(
            embedding0, self.cluster_center, self.args)

        target = get_auxiliary_distribution(cluster_prob).detach()
     
        return_dic['cluster_prob'] = cluster_prob
        return_dic['cluster_prob1'] = cluster_prob1
        return_dic['cluster_prob2'] = cluster_prob2
        return_dic['target'] = target

        return return_dic
