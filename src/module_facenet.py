import sys
sys.path.append('facenet-pytorch-glint5360k')

import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from datasets.LFWDataset import LFWDataset
from losses.triplet_loss import TripletLoss
from datasets.TripletLossDataset import TripletFaceDataset
from validate_on_LFW import evaluate_lfw
from plot import plot_roc_lfw, plot_accuracy_lfw
from tqdm import tqdm
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from models.mobilenetv2 import MobileNetV2Triplet
from models.resnet import (
    Resnet18Triplet,
    Resnet34Triplet,
    Resnet50Triplet,
    Resnet101Triplet,
    Resnet152Triplet
)

def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "mobilenetv2":
        model = MobileNetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    print("Using {} model architecture.".format(model_architecture))

    return model


# Instantiate model
model = set_model_architecture(
    model_architecture=model_architecture,
    pretrained=pretrained,
    embedding_dimension=embedding_dimension
)

# load to GPU

# load chkpt


checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['model_state_dict'])



embeddings = model(imgs)