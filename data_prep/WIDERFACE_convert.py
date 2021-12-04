import torch
import PIL
import os
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import IPython.display as ipd
import cv2
import numpy as np

import pickle
from tqdm import tqdm

def generate_label(path_gt,root_img,root_out):

    state_read = 0
    cnt = 0
    input_w = 224
    input_h = 224
    n_face = 0
    cnt_all = 0
    with open(path, 'r') as f:
        data = {}
        for line in tqdm(f):
            prev_line = line 
            #print("line("+str(state_read)+") : " + line )
            if state_read == 0 : 
                name = line.split('\n')[0]
                #print(name)
                state_read = 1
                cnt_all +=1
                data[name]={}
                detected = 0 
            elif state_read == 1 :
                num_face = int(line)
                num2read = num_face
                if num_face == 0 :
                    state_read = -1
                else :
                    data[name]["num_face"] = num_face
                    data[name]["faces"]=[]
                    state_read = 2
            elif state_read == -1 :
                state_read = 0
            elif state_read == 2 :
                anno = line
                anno = anno.split()
                #print(anno)
                x1 = int(anno[0])
                y1 = int(anno[1])
                w = int(anno[2])
                h = int(anno[3])
                blur = int(anno[4])
                expression = int(anno[5])
                illumination = int(anno[6])
                invalid = int(anno[7])
                occlusion = int(anno[8])
                pose = int(anno[9])
                num2read-=1
                
                if not (w >= 50 and h >= 50 and not invalid) :
                    pass
                else : 
                    data[name]["faces"].append([x1,y1,w,h])
                    detected +=1
                    n_face += 1
                
                if num2read <= 0 :
                    state_read = 0
                    if detected > 0 :
                        cnt+=1
                    else :
                        del data[name]
                    detected = 0
            else :
                raise Exception('Unknown state_read : ' + str(state_read))

    with open(root_output+'/label.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(str(cnt) +'/'+str(cnt_all))
    print(n_face)

if __name__ == '__main__':
    for category in ['train','val'] : 
        path = '/home/data2/kbh/WIDER_FACE/wider_face_split/wider_face_'+category+'_bbx_gt.txt'
        root_img = '/home/data2/kbh/WIDER_FACE/WIDER_'+category + '/images/'
        root_output = '/home/data2/kbh/WIDER_FACE/WIDER_'+category+'/'
        generate_label(path,root_img,root_output)