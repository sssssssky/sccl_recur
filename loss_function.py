import torch
import torch.nn as nn

class instance_CL_loss(nn.Module):
    def __init__(self, tempreture):
        super(instance_CL_loss, self).__init__()
        self.tempreture = tempreture
        self.eps = 1e-8
        
    def forward(self, feat1, feat2):
        all_feature = torch.cat([feat1, feat2], dim = 0)
        dot_contrast = torch.div(
            torch.matmul(all_feature, all_feature.T),
            self.tempreture)
        dot_contrast = torch.exp(dot_contrast)

        first_floor = torch.cat([torch.zeros(feat1.shape[0], feat1.shape[0]),torch.eye(feat1.shape[0], feat1.shape[0])], dim = 1)
        second_floor = torch.cat([torch.eye(feat1.shape[0], feat1.shape[0]), torch.zeros(feat1.shape[0], feat1.shape[0])], dim = 1)
        numerator_mask = torch.cat([first_floor, second_floor], dim = 0).cuda()
        
        denominator_mask = torch.ones_like(dot_contrast) - torch.eye(dot_contrast.shape[0]).cuda() - numerator_mask

        numerator = dot_contrast * numerator_mask
        denominator = dot_contrast * denominator_mask
    
        constrast_loss = -8* torch.log(numerator.sum(1) / denominator.sum(1))
        #constrast_loss = denominator.max(1)[0]/numerator.sum(1)

        return  constrast_loss.mean()
    