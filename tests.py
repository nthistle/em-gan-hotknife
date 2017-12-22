from keras.layers import Conv3D, Conv3DTranspose, UpSampling3D, Dense, Reshape, Flatten, Activation, Input
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
from util import *

def random_noise_gen_with_mask(mask, shape=(20,20,20)):
	while True:
		n = np.array([np.random.random(shape + (1,))])
		n2 = n.copy()
		n2[:,11:19,:,:,:] = 0.0 # zeros
		yield np.array([np.rollaxis(np.array([n[...,0],mask]), 0, len(shape)+2), n2])


def random_noise_gen(shape=(20,20,20)):
	while True:
		n = np.array([np.random.random(shape + (1,))])
		n2 = n.copy()
		n2[:,11:19,:,:,:] = 0.0 # zeros
		yield np.array([n, n2])

## Test the masked mean squared error loss

def test_masked_mse_loss(shape=(30,30,30)):
	return train_model(get_model(shape), shape)

def get_model(shape=(30,30,30), mask=None):
	if mask is None:
		mask = get_standard_mask(shape=shape, cut_width=20)

	model = Sequential()

	model.add(Conv3D(16, (3,3,3), padding="same", input_shape=(shape+(2,))))
	model.add(LeakyReLU(0.2))

	model.add(Conv3D(8, (3,3,3), padding="same"))
	model.add(LeakyReLU(0.2))

	model.add(Conv3D(1, (1,1,1), padding="same", input_shape=(shape+(1,)), activation="relu"))
	###model.add(Conv3D(1, (1,1,1), padding="same", activation="relu"))

	model.summary()

	loss_func = get_mse_masked_loss(mask)

	model.compile(Adam(), loss=loss_func, metrics=["mae"])
	
	return model


def train_model(model, shape=(30,30,30)):

	#model.add(Conv3D(16, (3,3,3), padding="same", input_shape=(shape+(1,))))
	#model.add(LeakyReLU(0.2))

	#model.add(Conv3D(16, (3,3,3), padding="same"))
	#model.add(LeakyReLU(0.2))

	#model.add(Conv3D(8, (3,3,3), padding="same"))
	#model.add(LeakyReLU(0.2))

	model.fit_generator(random_noise_gen(shape=shape), 100, 2)

	return model