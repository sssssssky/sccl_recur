import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans


def get_kmean_center(model, dataloader: DataLoader, args: argparse.ArgumentParser):
    """get the label and cluster center of kmeans

    Args:
        model: PTM
        dataloader: Dataloader
        args: global args
    
    Returns:
        cluster_center
        predict_label

    """
    print('get kmean center.....')
    for i, batch in enumerate(dataloader):
        text, label = batch['text'], np.array(batch['label'])
        embedding_text = model.encode(
            text, show_progress_bar=False, convert_to_tensor=True).cpu().numpy()
        if i == 0:
            all_embedding_text = embedding_text
            all_label = label
        else:
            all_embedding_text = np.concatenate(
                (all_embedding_text, embedding_text), axis=0)
            all_label = np.concatenate((all_label, label))

    kmeans_float = KMeans(n_clusters=args.num_classes)
    kmeans_float.fit(all_embedding_text)

    return kmeans_float.cluster_centers_, kmeans_float.labels_


def get_cluster_prob(embedding, cluster_center, args: argparse.ArgumentParser):
    """get the  


    """
    origin_distance = torch.sum(
        (embedding.unsqueeze(1) - cluster_center)**2, dim=2)
    distance = (origin_distance/args.alpha + 1)**(-1*(args.alpha + 1)/2)
    prob = distance / torch.sum(distance, dim=1,  keepdim=True)
    return origin_distance.min(1)[0], prob


def get_centers_distance(cluster_center):

    cluster_center = cluster_center.cpu()
    ab = torch.matmul(cluster_center, cluster_center.T)
    aa = ab.diag().reshape(-1, 1)*torch.ones(ab.shape[0])
    bb = torch.ones(ab.shape[0]).reshape(-1, 1)*ab.diag()
    distance = aa + bb - 2*ab
    return distance.sum()/2


def get_auxiliary_distribution(cluster_prob):
    """

    """
    fk = torch.sum(cluster_prob, dim=0)
    qjk = (cluster_prob ** 2)/fk
    auxiliary_distribution = (qjk.t() / torch.sum(qjk, dim=1)).t()

    return auxiliary_distribution
