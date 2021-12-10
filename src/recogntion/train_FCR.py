import os
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics

from Dataset_FCR import Dataset_FCR
from FCR import FCR

from tqdm import tqdm

## params
device = "cuda:1"
torch.cuda.set_device(device)
print(device)

batch_size = 64
num_epochs = 30
num_workers = 32
root = "../../DB/"

best_loss = 1e6

## data
dataset_train = Dataset_FCR(root, is_train=True)
dataset_test  = Dataset_FCR(root, is_train=False)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

print('len train loader : '+str(len(loader_train)))
print('len test loader : '+str(len(loader_test)))

## model
model = FCR()
model = model.to(device)


criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    factor=0.5,
    patience=2)

step = 0
for epoch in range(num_epochs):
    model.train()
    loss_train = 0

    ## Train    
    for idx, data in enumerate(loader_train):
        step += 1
        ## img = [n_batch, n_mels, n_frame]

        feat = data["feat"].to(device)
        label = data["label"].to(device)
        output = torch.squeeze(model(feat))

        loss = criterion(output.float(), label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % 100 == 1:
            print("TRAIN:: Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, idx+1,len(loader_train), loss.item()))

    ## Eval
    model.eval()
    with torch.no_grad():
        val_loss = 0.0

        for j, data in enumerate(loader_test):

            feat = data["feat"].to(device)
            label = data["label"].to(device)
            output =torch.squeeze(model(feat))

            loss = criterion(output.float(), label.float())
            val_loss +=loss.item()

        val_loss = val_loss/len(loader_test)
        scheduler.step(val_loss)

        print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(loader_test), val_loss))

    torch.save(model.state_dict(), "lastmodel.pt")
    if best_loss >  val_loss:
        torch.save(model.state_dict(), "bestmodel.pt")
        best_loss = val_loss