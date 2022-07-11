import torch
from torch import nn
import sys
from model_irse import Backbone
import torch.nn.functional as F
model_paths = {
	'ir_se50': './pretrained_ckpts/model_ir_se50.pth',
}

class IDFeatureNet(nn.Module):
    def __init__(self):
        super(IDFeatureNet, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x):
        return self.extract_feats(x)
