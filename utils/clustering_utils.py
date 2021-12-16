import torch
import numpy as np
from sklearn.cluster import KMeans


def get_kmean_center(model, dataloader, args):
    all_embedding_text, all_label = np.array([]), np.array([])
    for i, batch in enumerate(dataloader):
        text, label = batch['text'], np.array(batch['label'])
        embedding_text = model.encode(text)
        all_embedding_text = np.concatenate((all_embedding_text, embedding_text), axis = 1)
        all_label = np.concatenate((all_label, label), axis= 1)
    
    kmean_model = KMeans(args.num_classes)
    kmean_model.fit(all_embedding_text)

    return kmean_model.cluster_centers_

    
        

