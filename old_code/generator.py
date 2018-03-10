from keras.layers import Conv3D, Conv3DTranspose, UpSampling3D, Dense, Reshape, Flatten, Activation, Input, MaxPooling3D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
from util import *


## 64x64x64
def get_generator_valid(shape):
	inp = Input(shape=(shape + (1,)))  # 64^3
	layer = Conv3D(8, (3,3,3), activation="relu")(inp) # 62^3
	layer = Conv3D(16, (3,3,3), activation="relu")(layer) # 60^3
	layer = MaxPooling3D((2,2,2))(layer) # 30^3
	layer = Conv3D(32, (3,3,3), activation="relu")(layer) # 28^3
	layer = Conv3D(48, (3,3,3), activation="relu")(layer) # 26^3
	layer = MaxPooling3D((2,2,2))(layer) # 13^3
	layer = Conv3D(64, (3,3,3), activation="relu")(layer) # 11^3
	layer = UpSampling3D((2,2,2))(layer) # 22^3
	layer = Conv3D(48, (3,3,3), activation="relu")(layer) # 20^3
	layer = Conv3D(32, (3,3,3), activation="relu")(layer) # 18^3
	layer = UpSampling3D((2,2,2))(layer) # 36^3
	layer = Conv3D(16, (3,3,3), activation="relu")(layer) # 34^3
	layer = Conv3D(8, (3,3,3), activation="relu")(layer) # 32^3
	layer = Conv3D(1, (1,1,1), activation="sigmoid")(layer) #32^3
	decoded = layer
	model = Model(inp, decoded)
	return model

## Generator has an Autoencoder architecture, maps (32,)^3 to 8092-dimensional
## latent space, then back to (32,)^3
def get_generator(shape):
	inp = Input(shape=(shape + (1,)))
	layer = Conv3D(8, (3,3,3), activation="relu", padding="same")(inp)
	layer = Conv3D(16, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer)
	encoded = layer
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer)
	layer = UpSampling3D((2,2,2))(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = UpSampling3D((2,2,2))(layer)
	layer = Conv3D(16, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(8, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(1, (3,3,3), activation="sigmoid", padding="same")(layer)
	decoded = layer
	model = Model(inp, decoded)
	return model



def get_generator_bad(shape):
	inp = Input(shape=(shape + (1,)))
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(inp)
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(16, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(8, (3,3,3), activation="relu", padding="same")(layer)
	encoded = layer
	layer = Conv3D(16, (3,3,3), activation="relu", padding="same")(layer)
	layer = UpSampling3D((2,2,2))(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = UpSampling3D((2,2,2))(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(1, (3,3,3), activation="sigmoid", padding="same")(layer)
	decoded = layer
	model = Model(inp, decoded)
	return model