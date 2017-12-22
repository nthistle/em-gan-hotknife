from keras.layers import Conv3D, Conv3DTranspose, UpSampling3D, Dense, Reshape, Flatten, Activation, Input, MaxPooling3D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
from util import *


## Discriminator is similar to first half of generator, reduces to features
## but then it has Dense layers
def get_discriminator(shape):
	inp = Input(shape=(shape + (1,)))
	layer = Conv3D(16, (3,3,3), activation="relu", padding="same")(inp)
	layer = Conv3D(16, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(32, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer)
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer)
	layer = MaxPooling3D((2,2,2), padding="same")(layer)
	layer = Conv3D(64, (3,3,3), activation="relu", padding="same")(layer) 
	# at this point it's 64 4x4x4 filters, consider doing fewer of larger
	layer = Reshape((4096,))(layer)
	layer = Dense(16, activation="relu")(layer)
	layer = Dense(1, activation="sigmoid")(layer)
	model = Model(inp, layer)
	return model