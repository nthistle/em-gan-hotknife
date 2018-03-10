from keras.layers import Conv3D, Conv3DTranspose, UpSampling3D, Dense, Reshape, Flatten, Activation, Input, MaxPooling3D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

from util import *

import numpy as np

def get_generator(relu_leak=0.2):
	"""Returns valid-padded small generator, mapping 64x64x64 input to 32x32x32 output.
	Architecture is similar to an autoencoder; no skip connections.

	Keyword arguments:
	relu_leak -- Alpha parameter to the LeakyReLU layers
	"""
	input_layer = Input(shape = (64,64,64,1))

	conv1_1 = Conv3D(8, (3,3,3))(input_layer)
	relu1_1 = LeakyReLU(relu_leak)(conv1_1)
	conv1_2 = Conv3D(16, (3,3,3))(relu1_1)
	relu1_2 = LeakyReLU(relu_leak)(conv1_2)

	pool1 = MaxPooling3D((2,2,2))(relu1_2)

	conv2_1 = Conv3D(32, (3,3,3))(pool1)
	relu2_1 = LeakyReLU(relu_leak)(conv2_1)
	conv2_2 = Conv3D(48, (3,3,3))(relu2_1)
	relu2_2 = LeakyReLU(relu_leak)(conv2_2)

	pool2 = MaxPooling3D((2,2,2))(relu2_2)

	conv3 = Conv3D(64, (3,3,3))(pool2)
	relu3 = LeakyReLU(relu_leak)(conv3)

	upsamp1 = UpSampling3D((2,2,2))(relu_3)

	conv4_1 = Conv3D(48, (3,3,3))(upsamp1)
	relu4_1 = LeakyReLU(relu_leak)(conv4_1)
	conv4_2 = Conv3D(32, (3,3,3))(relu4_1)
	relu4_2 = LeakyReLU(relu_leak)(conv4_2)

	upsamp2 = UpSampling3D((2,2,2))(relu4_2)

	conv5_1 = Conv3D(16, (3,3,3))(upsamp2)
	relu5_1 = LeakyReLU(relu_leak)(conv5_1)
	conv5_2 = Conv3D(8, (3,3,3))(relu5_1)
	relu5_2 = LeakyReLU(relu_leak)(conv5_2)

	conv5_3 = Conv3D(1, (1,1,1), activation="sigmoid")(relu5_2)
	output_layer = conv5_3

	return Model(input_layer, output_layer)


def get_discriminator(relu_leak=0.2):
	"""Returns small discriminator, mapping 32x32x32 into a single output prediction.
	Architecture is fairly standard, two layers of convolutions between max pooling layers.

	Keyword arguments:
	relu_leak -- Alpha parameter to the LeakyReLU layers
	"""
	input_layer = Input(shape = (32,32,32,1))

	conv1_1 = Conv3D(16, (3,3,3))(input_layer)
	relu1_1 = LeakyReLU(relu_leak)(conv1_1)
	conv1_2 = Conv3D(32, (3,3,3))(relu1_1)
	relu1_2 = LeakyReLU(relu_leak)(conv1_2)

	pool1 = MaxPooling3D((2,2,2))(relu1_2)

	conv2_1 = Conv3D(32, (3,3,3))(pool1)
	relu2_1 = LeakyReLU(relu_leak)(conv2_1)
	conv2_2 = Conv3D(64, (3,3,3))(relu2_1)
	relu2_2 = LeakyReLU(relu_leak)(conv2_2)

	pool2 = MaxPooling3D((2,2,2))(relu2_2)

	conv3_1 = Conv3D(64, (3,3,3))(pool2)
	relu3_1 = LeakyReLU(relu_leak)(conv3_1)
	conv3_2 = Conv3D(128, (1,1,1))(relu3_1)
	relu3_2 = LeakyReLU(relu_leak)(conv3_2)

	flat1 = Flatten()(relu3_2)

	dense1 = Dense(32)(flat1)
	relu4 = LeakyReLU(relu_leak)(dense1)

	dense2 = Dense(1, activation="sigmoid")(relu4)
	output_layer = dense2

	return Model(input_layer, output_layer)