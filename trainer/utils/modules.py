from matplotlib.pyplot import axis
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class datiCliniciLateFusion(nn.Module):
    def __init__(self, net_img, fusion_dim, in_dim_datiClinici, num_classes):
        super(datiCliniciLateFusion, self).__init__()

        self.net_img = net_img
        self.fc_datiClinici = nn.Linear(in_dim_datiClinici, fusion_dim)

        self.classifier_label = nn.Linear(2*fusion_dim, num_classes)
        
    def forward(self, imgs, datiClinici):
        o_img = self.net_img(imgs)
        o_datiClinici = self.fc_datiClinici(datiClinici)

        o = self.classifier_label(torch.cat( (o_img, o_datiClinici) , dim=1))
        return o