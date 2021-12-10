import os,glob

import torch
import numpy as np
import sys

import cv2
from tqdm import tqdm

path_add = "./yolo5face"
sys.path.append(path_add)
from module_detect import YOLO5face_detector
for st in sys.path :
    if path_add in st :
        sys.path.remove(st)
        
path_add = "./insightface/recognition/arcface_torch"
sys.path.append(path_add)
from module_embedding import module_embedding
for st in sys.path :
    if path_add in st :
        sys.path.remove(st)

root_pretrained = "../pretrained/"
device = "cuda:0"
loss = "cosface"
#loss = "arcface"
name_model =  "yolov5n-0.5.pt"

detector = YOLO5face_detector(root_pretrained+name_model,device)
embedding = module_embedding(device,loss=loss)

list_target = glob.glob(os.path.join("../data","*"))

root_output = "../DB"

for path in tqdm(list_target) :
    vid = cv2.VideoCapture(path)

    file_name = path.split('/')[-1]
    name = file_name.split('.')[0]
    id= name.split('_')[0]

    os.makedirs(os.path.join(root_output,loss),exist_ok=True)
    os.makedirs(os.path.join(root_output,loss,id),exist_ok=True)

    cnt = 0
    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret :
            xywh = detector.detect(frame)
            if xywh is not None : 
                crop = frame[xywh[1]:xywh[3],xywh[0]:xywh[2]]
                crop = cv2.resize(crop, dsize=(112,112))

                t_feat = embedding.embed(crop)
                while os.path.isfile(os.path.join(root_output,loss,id,id+"_"+str(cnt)+".npy")) :
                    cnt+=1
                t_feat = np.squeeze(t_feat)
                np.save(os.path.join(root_output,loss,id,id+"_"+str(cnt)+".npy"),t_feat)
                
                #print(feat.shape)
            else :
                pass
        else :
            break
    vid.release()
    
    cnt = 0 
    feat = None
    list_feat = glob.glob(os.path.join(root_output,loss,id,"*_*.npy"))
    for path2 in list_feat : 
        t_feat = np.load(path2)
        t_feat = np.expand_dims(t_feat,0)
        if feat is None :
            feat = t_feat
        else :
            feat = np.concatenate((feat,t_feat),0)
        cnt+=1

    feat = feat.mean(0)
    np.save(os.path.join(root_output,loss,id,id+".npy"),feat)
    #print(name + " | " + str(cnt))