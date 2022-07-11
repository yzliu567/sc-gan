import numpy as np
import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from torchvision import utils
from PIL import Image
import hopenet
if torch.__version__ >= "1.4.0":
    from kornia.enhance import Normalize
else:
    from kornia.color import Normalize

class GetYaw:
    def __init__(self, device="cuda"):
        self.snapshot_path = './pretrained_ckpts/hopenet_robust_alpha1.pkl'
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.saved_state_dict = torch.load(self.snapshot_path, map_location=device)
        self.model.load_state_dict(self.saved_state_dict)
        self.device = device
        if device == "cuda":
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(device)
        self.transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.norm = Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

    def path_to_yaw(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        #print(np.asarray(img))
        img = self.transformations(img)
        #print(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).to(self.device)

        yaw, pitch, roll = self.model(img)
        yaw_predicted = F.softmax(yaw, dim=1)
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 3 - 99

        return yaw_predicted

    def tensor_to_yaw(self, img):
        #img: B * 3 * 256 * 256
        img = F.interpolate(img, size=(224,224), mode='bilinear')
        img = torch.clamp(img, -1, 1)
        img = (img + 1) * 0.5
        img = self.norm(img)

        yaw, pitch, roll = self.model(img)
        yaw_predicted = F.softmax(yaw, dim=1)
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * 3 - 99

        return yaw_predicted

if __name__ == '__main__':
    testmodel = GetYaw()
    im_path = 'xxx.jpg'
    print(testmodel.path_to_yaw(im_path))

    pt_path = 'xxx.pt'
    img = torch.load(pt_path)
    print(testmodel.tensor_to_yaw(img))