import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import List
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter
from sklearn.metrics.cluster import normalized_mutual_info_score

from dataloader import get_dataloader
from utils.utils import *
from utils.clustering_utils import get_cluster_prob, get_centers_distance
from evalution import evalution
from model import sccl
from loss_function import instance_CL_loss


def eval_step(sccl_model: sccl, args: argparse.ArgumentParser, writer: SummaryWriter, j: int):
    print("Step: {}   start eval......".format(j))
    sccl_model.eval()
    eval_dataloader = get_dataloader(args)
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            ids, y_true = sccl_model.char2id(batch)
            embedding_text = sccl_model.id2embedding(ids[0])
            min_distance, cluster_prob = \
                get_cluster_prob(
                                    embedding_text,
                                    sccl_model.cluster_center,
                                    args
                                 )

            if i != 0:
                all_embedding_text = np.concatenate((all_embedding_text, embedding_text.cpu().numpy()), axis = 0)
                all_label = np.concatenate((all_label, y_true))
                all_pre = np.concatenate((all_pre,cluster_prob.max(1)[1].cpu().numpy()))
                all_min_distance = np.concatenate((all_min_distance, min_distance.cpu().numpy()))
            else:
                all_embedding_text = embedding_text.cpu().numpy()
                all_label = y_true
                all_pre = cluster_prob.max(1)[1].cpu().numpy()
                all_min_distance = min_distance.cpu().numpy()
  
    kmeans = KMeans(n_clusters=args.num_classes, random_state=args.seed)
    kmeans.fit(all_embedding_text)
    pred_labels = kmeans.labels_.astype(np.int32)
    
    cluster_density = kmeans.inertia_
    cluster_centers_distance = get_centers_distance(torch.tensor(kmeans.cluster_centers_))
    print('[cluster] density: {}'.format(cluster_density))
    print('[cluster] center distance： {}'.format(cluster_centers_distance.cpu().numpy()))
    
    matrix = evalution(args.num_classes)
    kmean_purity = matrix.purity(all_label, pred_labels)
    matrix.get_cluster_label(all_label,pred_labels)
    kmean_acc = matrix.acc()
    print("[cluster] accuracy: {}".format(kmean_acc))
    nmi = normalized_mutual_info_score(all_label, pred_labels)
    print('[cluster] nmi:{}'.format(nmi))
    
    writer.add_scalars('eval/kmean', {"kmean_purity": kmean_purity, "kmean_acc": kmean_acc, "kmean_nmi":nmi}, j)

    model_density = all_min_distance.sum()
    model_center_distance = get_centers_distance(sccl_model.cluster_center.detach())
    print('[model] density: {}'.format(model_density))
    print('[model] center distance： {}'.format(model_center_distance.cpu().numpy()))
    
    
    matrix = evalution(args.num_classes)
    train_purity = matrix.purity(all_label, all_pre)
    matrix.get_cluster_label(all_label, all_pre)
    train_acc = matrix.acc()
    print("[model] accuracy: {}".format(train_acc))
    nmi = normalized_mutual_info_score(all_label, all_pre)
    print('[model] nmi:{}'.format(nmi))
    
    writer.add_scalars('eval/model_', {"model_purity": train_purity, "model_acc": train_acc, "model_nmi":nmi}, j)
    
    writer.add_scalars('density',{'model_density':model_density,'kmean_density':cluster_density},j)
    writer.add_scalars('centers_distance',{'model_centers_distance': cluster_centers_distance,'kmean_centers_distance': model_center_distance},j)
    
    return kmeans.cluster_centers_


def train(args: argparse.ArgumentParser):
    set_global_random_seed(args)
    writer = SummaryWriter(comment= args.train_name, filename_suffix = args.train_name)

    sccl_model = sccl(args).cuda()
    sccl_model.train()
    optimizer = torch.optim.Adam([
            {'params': sccl_model.embedding_model_without_pool.parameters()}, 
            {'params': sccl_model.instance_cl.parameters(), 'lr': args.lr * args.lr_scale},
            {'params': sccl_model.cluster_center, 'lr': args.lr * args.lr_scale}], lr=args.lr)  
                
    constrast_loss = instance_CL_loss(args.tempreture)
    cluster_loss = nn.KLDivLoss(size_average=False)

    print(optimizer)
    dataloader = get_dataloader(args,1000)
    for j,batch in enumerate(dataloader):
        
        if(j == 0):
            eval_step(sccl_model, args, writer, j)
        output  = sccl_model.forward(batch)
        
        constrast = constrast_loss(output['feat1'], output['feat2'])
        clustering = cluster_loss((output['cluster_prob'] + 1e-08).log(),output['target']) / output['cluster_prob'].shape[0]
        loss = constrast + clustering
        
        if args.is_use_consistancy:
            consistancy_loss = cluster_loss((output['cluster_prob1'] + 1e-08).log(),output['cluster_prob']) / output['cluster_prob'].shape[0] \
                                + cluster_loss((output['cluster_prob2'] + 1e-08).log(),output['cluster_prob']) / output['cluster_prob'].shape[0]
            loss = loss +consistancy_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar('loss/Instance-CL_loss',constrast.item() , j)
        writer.add_scalar('loss/clustering_loss',clustering.item() , j)
        if (j % args.eval_step ==  0) & (j != 0):
            print('\n')
            new_cluster_center = eval_step(sccl_model, args, writer, j)
            if args.is_change_centers:
                sccl_model.reset_cluster_center(new_cluster_center)
                sccl_model.cuda()
            sccl_model.train()
        if j > args.max_step:
            break

    writer.close()


def get_argparser(argv: List) ->argparse.ArgumentParser:
    """get global args

    Args:
        argv: sys.argv[1:]

    Returns:
        args

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_name', type = str, default = 'googlenewsTS_contextual_change_center(400)')
    parser.add_argument('--train_data_path', type = str, default = 'train_data/googlenewsTS_contextual_augment.csv', help = 'the path of train data')
    parser.add_argument('--batch_size', type = int,default = 350, help = 'batch size')
    parser.add_argument('--gpu', type = str, default = 'gpu', help = 'cpu or gpu')
    parser.add_argument('--seed', type = int, default = 0, help = 'the global seed')
    parser.add_argument('--model', type = str ,default = 'distil' ,help = 'the pre_train language model' )
    parser.add_argument('--max_len', type = int, default = 32, help = 'the max lenght of sentence')
    parser.add_argument('--num_classes', type = int, default= 152 ,help = 'the class num of news or text')
    parser.add_argument('--tempreture', type = float, default = 0.5, help = 'the tempreture of  instance_CL_loss')
    parser.add_argument('--alpha', type = int, default = 1, help = 'the degree of freedom of the Students t-distribution')
    parser.add_argument('--lr', type = float, default = 1e-5, help = "lr")
    parser.add_argument('--lr_scale', type = float, default = 100.0, help = "lr_scale")
    parser.add_argument('--eval_step', type = int, default = 400, help = 'every n step to eval' )
    parser.add_argument('--max_step', type = int, default = 3000, help = 'the max step to train' )
    parser.add_argument('--is_use_consistancy', type = bool, default = False, help = 'is use consistancy' )
    parser.add_argument('--is_change_centers', type = bool, default = True, help = 'is change_centers' )
    args = parser.parse_args(argv)
    return args


if __name__ =='__main__':
    args = get_argparser(sys.argv[1:])
    train(args)




    