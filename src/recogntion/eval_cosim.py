## evaluation by cosine similarity

from scipy import spatial
import torch
import os,glob
import numpy as np

from tqdm import tqdm

root = "../../DB/"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--loss',type=str,default="arcface")
parser.add_argument('--n_iter_intra',type=int,default=500)
parser.add_argument('--n_iter_inter',type=int,default=5000)
parser.add_argument('--n_frame',type=int,default=10)
parser.add_argument('--thr',type=float,default=0.4)
args = parser.parse_args()

loss = args.loss
n_iter_intra = args.n_iter_intra
n_iter_inter = args.n_iter_inter

n_frame = args.n_frame
thr = args.thr

list_target = glob.glob(os.path.join(root,loss,"*","*_*.npy"))
dict_target = {}
for path in list_target :
    name = path.split('/')[-1]
    id = name.split('_')[0]
    if id in dict_target : 
        dict_target[id].append(path)
    else :
        dict_target[id]=[path]

idx = np.random.randint(len(list_target))
i_path = list_target[idx]
i_name = i_path.split('/')[-1]
i_id = i_name.split('_')[0]
i_path = os.path.join(root,loss,i_id,i_id+".npy")

## intra
intra_acc = 0
for i in tqdm(range(n_iter_intra)) :
    o_id = i_id
    o_idx = np.random.randint(len(dict_target[i_id]) - n_frame)
    o_path = dict_target[i_id][o_idx]

    i_np = torch.from_numpy(np.load(i_path))

    cosim = 0
    for j in range(n_frame):
        o_np = torch.from_numpy(np.load(dict_target[o_id][o_idx]))
        cosim += 1 - spatial.distance.cosine(i_np, o_np)
    cosim = cosim/n_frame

    if cosim >= thr :
        intra_acc +=1
    else :
        pass
    #print(i_id + " | " + o_id + " | " + str(cosim))

intra_acc  = intra_acc / n_iter_intra

## inter 
inter_acc = 0
for i in tqdm(range(n_iter_inter)) :
    o_id = i_id
    while o_id == i_id :
        o_id = list(dict_target.keys())[np.random.randint(len(list(dict_target.keys())))]
    o_idx = np.random.randint(len(dict_target[o_id]) - n_frame) 
    o_path = dict_target[o_id][o_idx]

    i_np = torch.from_numpy(np.load(i_path))

    cosim = 0
    for j in range(n_frame):
        o_np = torch.from_numpy(np.load(dict_target[o_id][o_idx]))
        cosim += 1 - spatial.distance.cosine(i_np, o_np)
    cosim = cosim/n_frame

    if cosim >= thr :
        pass
    else :
        inter_acc += 1
    #print(i_id + " | " + o_id + " | " + str(cosim))
    

inter_acc = inter_acc / n_iter_inter
print('----- loss : '+ str(loss)+' | n_frame : ' + str(n_frame) +" | thr : "+str(thr)+' ----')
print('intra : ' + str(intra_acc))
print('inter : ' + str(inter_acc))
