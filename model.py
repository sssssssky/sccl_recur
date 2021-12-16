import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from utils.clustering_utils import get_kmean_center
from utils.utils import char2id, get_model_name, id2embeding
from loss_function import instance_CL_loss, cluster_loss


class sccl(nn.Module):
    def __init__(self, dataloader, args):

        self.embedding_model = SentenceTransformer(get_model_name(args))
        self.args = args
        self.dataloader = dataloader
        self.tokenizer = self.embedding_model[0].tokenizer
        self.embedding_model_without_pool = self.embedding_model[0].auto_model
        self.embedding_size = self.embedding_model_without_pool.hidden_size
        
        
        self.instance_cl = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(inplace = True),
            nn.Linear(self.embedding_size, 128)
        )
        self.constrast_loss = instance_CL_loss(args.tempreture)

        initial_cluster_center = torch.tensor(
            get_kmean_center(self.embedding_model, dataloader,self.args),dtype = torch.float , requires_grad = True
        )
        self.cluster_center = nn.parameter(initial_cluster_center)
        self.cluster_loss = cluster_loss(args.alpha)



    def char2id(self, batch):
        text, text1, text2 = batch['text'], batch['text1'], batch['text2']
        label =batch['label'] if self.args.gpu == 'cpu' else batch['label'].cuda()
    
        all_text = [text, text1, text2]
        feat = []
        for text_i in all_text:
            ids = self.tokenizer.batch_encode_plus(text_i, max_length = self.args.max_len, return_tenser = 'pt', \
                padding = 'longest', truncating = True)
            feat.append(ids)
        return ids,label

    def id2embedding(self, ids, pooling = 'mean'):
        out = self.embedding_model_without_pool.forward(**ids)
        attention_mask = ids['attention_mask'].unsqueeze(-1)
        cls_out = out[0]
        embedding = torch.sum(cls_out*attention_mask, dim = 1)/torch.sum(attention_mask, dim = 1)
        return embedding

    def forward(self, batch):
        ids = self.char2id(batch)
        embedding0, embedding1, embedding2 = self.id2embeding(ids[0]), self.id2embeding(ids[1]), self.id2embeding(ids[2])

        # instance_CL_loss
        feat1 = F.normalize(self.instance_cl(embedding1), dim=1)
        feat2 = F.normalize(self.instance_cl(embedding2), dim=1)
        constrast_loss = self.constrast_loss(feat1, feat2)
        
        loss = constrast_loss
        
        # clusterring_loss






