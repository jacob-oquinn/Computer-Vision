from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from Autoencoder import Autoencoder 
import argparse
import numpy as np 
    

def run_main(FLAGS):    
    # load in MNIST
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    
    # rescale to between 0-1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Initialize the model
    autoencoder = Autoencoder(FLAGS.mode)
    
    # train the model
    autoencoder.fit(x_train, x_train, epochs=10, batch_size=64, validation_split=.15)
    
    # predict using the model, 20 images, 2 per class
    fig, axis = plt.subplots(nrows=4, ncols=10)
    fig.set_size_inches(15,6)
    test_index = 0
    for i in range (4):
        for j in range(10):
            while(y_test[test_index] != j):
                test_index += 1
            axis[i,j].axis('off')
            if i%2==0:
                axis[i,j].imshow(x_test[test_index], cmap='gray')
            else:
                pred = autoencoder_conv.predict([x_test[test_index].reshape(-1,28,28,1)])[0]
                axis[i,j].imshow(pred, cmap='gray')
   
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Autoencoder Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-2.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    