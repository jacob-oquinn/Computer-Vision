import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
            
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        
    # Dense autoencoder
    def model_1(self, x):
        encoder_input = keras.Input(shape=(28,28,1), name='img')
        x = keras.layers.Flatten()(encoder_input)
        x = keras.layers.Dense(256)(x)
        encoder_output = keras.layers.Dense(128)(x)

        encoder = keras.Model(encoder_input, encoder_output, name='encoder')

        decoder_input = keras.layers.Dense(256)(encoder_output)
        x = keras.layers.Dense(784, activation='relu')(decoder_input)
        decoder_output = keras.layers.Reshape((28,28,1))(x)

        return decoder_output
        

    # Convolutional autoencoder
    def model_2(self, x):        
        encoder_input_conv = keras.Input(shape=(28,28,1), name='img')

        x = keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(encoder_input_conv)
        x = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)

        decoder_input_conv = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(encoder_output_conv)
        x = keras.layers.UpSampling2D(size=(2, 2))(decoder_input_conv)
        x = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        decoder_output_conv = keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu")(x)

        return decoder_output_conv

    
