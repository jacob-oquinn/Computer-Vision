import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
            
    # Gets the number of features in the matrix x
    def flatten_features(self, x):
        num_features = 1
        for s in x.size()[1:]:
            num_features *= s 
        return num_features
            
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        self.FC1 = nn.Linear(28*28, 100)
        self.FC2 = nn.Linear(40*18*18, 100)
        self.FC3 = nn.Linear(100, 100)
        self.FC_condense1 = nn.Linear(100, 10)
        
        self.FC4 = nn.Linear(40*18*18, 1000)
        self.FC5 = nn.Linear(1000, 1000)
        self.FC_condense2 = nn.Linear(1000, 10)
        
        self.convo1 = nn.Conv2d( 1, 40, 5)
        self.convo2 = nn.Conv2d(40, 40, 5)
        
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.5) 


        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
    # Baseline model. step 1
    def model_1(self, x):
        # ======================================================================
        # One fully connected layer.
        
        ## Create a fully connected (FC) hidden layer (with 100 neurons) 
        ## with sigmoid activation function. 
        # Train it with SGD
        # with a learning rate of 0.1 (a total of 60 epoch),
        # a mini-batch size of 10, 
        # and no regularization.
        
        # Reshape to (10, 28*28)
        x = x.view(-1, self.flatten_features(x))
        x = torch.sigmoid(self.FC1(x))
        x = self.FC_condense1(x)
        return x

    # Use two convolutional layers.
    def model_2(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        
        # Now insert two convolutional layers to the network built in STEP 1 
        # put pooling layer too for each convolutional layer 
        #    Pool over 2x2 regions, 40 kernels, stride =1, with kernel size of 5x5.
        x = self.pool(self.convo1(x))
        x = self.pool(self.convo2(x))
        
        x = x.view(-1, self.flatten_features(x))
        
        x = torch.sigmoid(self.FC2(x))
        x = self.FC_condense1(x)
        return x

    # Replace sigmoid with ReLU.
    def model_3(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        #: For the network depicted in STEP 2, replace Sigmoid with ReLU, and train the model with new
        #  learning rate (=0.03). Re-train the system with this setting.
        
        x = self.pool(self.convo1(x))
        x = self.pool(self.convo2(x))
        
        x = x.view(-1, self.flatten_features(x))
        
        x = self.relu(self.FC2(x))
        x = self.FC_condense1(x)
        
        return x

    # Add one extra fully connected layer.
    def model_4(self, x):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # Add another fully connected (FC) layer now (with 100 neurons) to the network built in STEP 3.
        # (remember that the first FC was put in STEP 1, here you are putting just another FC).
        
        x = self.pool(self.convo1(x))
        x = self.pool(self.convo2(x))
        
        x = x.view(-1, self.flatten_features(x))
        
        x = self.relu(self.FC2(x))
        x = self.relu(self.FC3(x))
        x = self.FC_condense1(x)
        
        return x

    # Use Dropout now.
    def model_5(self, x):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # Change the neurons numbers in FC layers into 1000. For regularization, use Dropout (with a rate of 0.5).
        # Train the whole system using 40 epochs.
        x = self.pool(self.convo1(x))
        x = self.pool(self.convo2(x))
        
        x = x.view(-1, self.flatten_features(x))
        x = self.dropout(x)
        x = torch.relu(self.FC4(x))
        x = self.dropout(x)
        x = torch.relu(self.FC5(x))
        x = self.FC_condense2(x)
        return x
    
    
