import torch
import torch.nn as nn



# Feature Extraction of YOLO-face, Based on Darknet53 implemented by devekioer0hye: https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class FeatureExtractionNetwork(nn.Module):
    def __init__(self, block):
        super(FeatureExtractionNetwork, self).__init__()

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=4)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=8)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out_s1 = self.residual_block3(out)

        out = self.conv5(out_s1)
        out_s2 = self.residual_block4(out)

        out = self.conv6(out_s2)
        out_s3 = self.residual_block5(out)


        return out_s1,out_s2,out_s3

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()

        self.out_1 = nn.Conv2d()
        self.up_1 = nn.Upsample()
        self.out_2 = nn.Conv2d()
        self.up_2 = nn.Upsample()
        self.out_3 = nn.Conv2d()

    def forward(self,s1,s2,s3):







class YOLO_face(nn.Module):
    def __init__(self):
        super(YOLO_face, self).__init__()
        
        self.feature_extraction = FeatureExtractionNetwork(DarkResidualBlock)


    def forward(self,x):
        feature = self.feature_extraction(x)
