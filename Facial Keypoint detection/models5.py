## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to
        # avoid overfitting
        
        #Conv block 1
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5)
        #self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout1 = nn.Dropout(p=0.1)
        
        #Conv block 2
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        #self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout2 = nn.Dropout(p=0.2)
        
        #Conv block 3
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        #self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout3 = nn.Dropout(p=0.3)
        
        #Conv block 4
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)
        #self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout4 = nn.Dropout(p=0.4)
        
        #Conv block 5
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout5 = nn.Dropout(p=0.5)
        
        #Fully connected layers
        self.fc1 = nn.Linear(512*5*5,1024)
        #self.bn5 = nn.BatchNorm2d(1000)
        self.dropout6 = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1024,136)
        #self.bn6 = nn.BatchNorm2d(512)
        #self.dropout5 = nn.Dropout(p=0.5)
        
        #nn.init.xavier_uniform(self.fc1.weight.data)
        #nn.init.xavier_uniform(self.fc2.weight.data)
        #nn.init.xavier_uniform(self.fc3.weight.data)
        
        
        
        
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.dropout1(self.pool1(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.elu(self.conv4(x))))
        x = self.dropout5(self.pool5(F.elu(self.conv5(x))))
        x = x.view(x.size(0),-1)
        x = self.dropout6(F.elu(self.fc1(x)))
        x = self.fc2(x)
   
        # a modified x, having gone through all the layers of your model, should be returned
        return x
