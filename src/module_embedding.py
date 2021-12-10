import cv2
import numpy as np
import torch

from backbones import get_model


class module_embedding():
    def __init__(self,device="cpu",loss="arcface"):
        self.net = get_model("r34",fp16=False)
        self.net.load_state_dict(torch.load("../pretrained/r34_"+loss+".pth"))
        self.net = self.net.to(device)
        self.net.eval()
        
        self.device = device

    @torch.no_grad()
    def embed(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255).sub_(0.5).div_(0.5)
        embedding = self.net(img)
        if self.device is not "cpu" : 
            embedding = embedding.to("cpu").numpy()
        else  :
            embedding = embedding.numpy()
        return embedding
