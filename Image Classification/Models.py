import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import exit

class Models(nn.Module):
            
    # Gets the number of features in the matrix x
    def flatten_features(self, x):
        num_features = 1
        for s in x.size()[1:]:
            num_features *= s 
        return num_features
            
    def __init__(self, mode):
        super(Models, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.dense1 = nn.Linear(8*8*3, 10)
    

        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=35, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=35, out_channels=100, kernel_size=3, padding=1)
        self.dense2_1 = nn.Linear(8*8*100, 300)
        self.dense2_2 = nn.Linear(300, 10)

        self.conv3_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dense3_1 = nn.Linear(4*4*100, 300)
        self.dense3_2 = nn.Linear(300, 10)

        if mode == 1:
            self.forward = self.model_1
        if mode == 2:
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
    # Just a Baseline model 
    def model_1(self, x):
        # Two Conv Layers

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.pool1(x)

        x = x.view(-1, self.flatten_features(x))
        x = self.dense1(x)
      
        x = torch.sigmoid(x)
        return x

    # Making Conv layers "thicker"
    def model_2(self, x):
        # Two Conv Layers
        # One Dense Layer

        x = self.conv2_1(x)
        x = self.pool1(x)
        x = self.conv2_2(x)
        x = self.pool1(x)

        x = x.view(-1, self.flatten_features(x))
        x = self.dense2_1(x)
        x = self.dense2_2(x)

        x = torch.sigmoid(x)
        return x

    # Making Conv layers "thicker"
    def model_3(self, x):
        # Two Conv Layers
        # One Dense Layer

        x = self.conv3_1(x)
        x = self.pool1(x)
        x = self.conv3_2(x)
        x = self.pool1(x)
        x = self.conv3_3(x)
        x = self.pool1(x)

        x = x.view(-1, self.flatten_features(x))
        x = self.dense3_1(x)
        x = self.dense3_2(x)

        x = torch.sigmoid(x)
        return x
