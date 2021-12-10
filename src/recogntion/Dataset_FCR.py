import torch
import os,glob
import numpy as np

id_train = [""]
id_test  = [""]

class Dataset_FCR(torch.utils.data.Dataset):
    def __init__(self,root,is_train=True):
        
        if is_train : 
            self.list_target = glob.glob(os.path.join(root,"cosface","train","*","*_*.npy"))
        else :
            self.list_target = glob.glob(os.path.join(root,"cosface","test","*","*_*.npy"))
        
        self.dict_target = {}
        for path in self.list_target :
            name = path.split('/')[-1]
            id = name.split('_')[0]
            if id in self.dict_target : 
                self.dict_target[id].append(path)
            else :
                self.dict_target[id]=[path]


    def __len__(self):
        return len(self.list_target)

    def __getitem__(self,idx):
        i_path = self.list_target[idx]
        i_name = i_path.split('/')[-1]
        i_id = i_name.split('_')[0]

        prob = np.random.rand(1)[0]
        # same
        if prob > 0.5 :
            o_idx = np.random.randint(len(self.dict_target[i_id])) 
            o_path = self.dict_target[i_id][o_idx]
            o_name = o_path.split('/')[-1]
            o_id = o_name.split('_')[0]
        # diff
        else :
            o_id = i_id
            while o_id == i_id :
                o_id = list(self.dict_target.keys())[np.random.randint(len(list(self.dict_target.keys())))]
            o_idx = np.random.randint(len(self.dict_target[o_id])) 
            o_path = self.dict_target[o_id][o_idx]

        if i_id == o_id :
            label = torch.tensor((1,0))
        else :
            label = torch.tensor((0,1))
        
        i_np = torch.from_numpy(np.load(i_path))
        o_np = torch.from_numpy(np.load(o_path))

        feat = torch.cat((i_np,o_np)).float()        

        data = {"feat":feat,"label":label}
        return data