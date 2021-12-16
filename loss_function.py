import torch
import torch.nn as nn

class instance_CL_loss(nn.Module):
    def __init__(self, tempreture):
        self.tempreture = tempreture
        
    def forward(self, feat1, feat2):
        



        constrast_loss = 0
        return  constrast_loss


class cluster_loss(nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, feat, cluster_center):


        clustering_loss = 0
        return clustering_loss

    