from keras.layers import Conv3D, Conv3DTranspose, UpSampling3D, Dense, Reshape, Flatten, Activation, Input, MaxPooling3D, Cropping3D, Concatenate, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import L1L2

import numpy as np

## 295 with factor=3 -> output size of 83
def get_generator_arch_a(init_filters=12, filter_scale=3, relu_leak=0.2, regularization=0.0):

	if regularization > 0.0:
		reg = lambda : L1L2(regularization, regularization)
	else:
		reg = lambda : None

	input_layer = Input(shape = (295,295,295,1))

	filter_count = init_filters
	stage1_in = input_layer
	conv1_1 = Conv3D(filter_count, (3,3,3), name="conv1_1", padding="valid", kernel_regularizer=reg())(stage1_in)
	relu1_1 = LeakyReLU(relu_leak, name="relu1_1")(conv1_1)
	conv1_2 = Conv3D(filter_count, (3,3,3), name="conv1_2", padding="valid", kernel_regularizer=reg())(relu1_1)
	relu1_2 = LeakyReLU(relu_leak, name="relu1_2")(conv1_2)
	## TODO ADD BN
	stage1_out = relu1_2

	pool1 = MaxPooling3D((3,3,3), name="pool1")(stage1_out)


	filter_count *= filter_scale
	stage2_in = pool1

	conv2_1 = Conv3D(filter_count, (3,3,3), name="conv2_1", padding="valid", kernel_regularizer=reg())(stage2_in)
	relu2_1 = LeakyReLU(relu_leak, name="relu2_1")(conv2_1)
	conv2_2 = Conv3D(filter_count, (3,3,3), name="conv2_2", padding="valid", kernel_regularizer=reg())(relu2_1)
	relu2_2 = LeakyReLU(relu_leak, name="relu2_2")(conv2_2)
	## TODO BATCH NORM
	stage2_out = relu2_2


	pool2 = MaxPooling3D((3,3,3), name="pool2")(stage2_out)


	filter_count *= filter_scale
	stage3_in = pool2

	conv3_1 = Conv3D(filter_count, (3,3,3), name="conv3_1", padding="valid", kernel_regularizer=reg())(stage3_in)
	relu3_1 = LeakyReLU(relu_leak, name="relu3_1")(conv3_1)
	conv3_2 = Conv3D(filter_count, (3,3,3), name="conv3_2", padding="valid", kernel_regularizer=reg())(relu3_1)
	relu3_2 = LeakyReLU(relu_leak, name="relu3_2")(conv3_2)
	## TODO BATCH NORM
	stage3_out = relu3_2

	pool3 = MaxPooling3D((3,3,3), name="pool3")(stage3_out)



	filter_count *= filter_scale
	stage4_in = pool3

	conv4_1 = Conv3D(filter_count, (3,3,3), name="conv4_1", padding="valid", kernel_regularizer=reg())(stage4_in)
	relu4_1 = LeakyReLU(relu_leak, name="relu4_1")(conv4_1)
	conv4_2 = Conv3D(filter_count, (3,3,3), name="conv4_2", padding="valid", kernel_regularizer=reg())(relu4_1)
	relu4_2 = LeakyReLU(relu_leak, name="relu4_2")(conv4_2)
	## TODO BATCH NORM
	stage4_out = relu4_2

	upsamp1 = UpSampling3D((3,3,3), name="upsamp1")(stage4_out)
	#upconv1 = Conv3D(64, (2,2,2), padding=padding, name="upconv1")(upsamp1)
	#uprelu1 = LeakyReLU(relu_leak, name="uprelu1")(upconv1)


	stage3_out_cropped = Cropping3D(cropping=6)(stage3_out)
	stage5_in = Concatenate()([upsamp1, stage3_out_cropped])
	filter_count //= filter_scale
	#stage5_in = upsamp1

	conv5_1 = Conv3D(filter_count, (3,3,3), name="conv5_1", padding="valid", kernel_regularizer=reg())(stage5_in)
	relu5_1 = LeakyReLU(relu_leak, name="relu5_1")(conv5_1)
	conv5_2 = Conv3D(filter_count, (3,3,3), name="conv5_2", padding="valid", kernel_regularizer=reg())(relu5_1)
	relu5_2 = LeakyReLU(relu_leak, name="relu5_2")(conv5_2)
	## TODO BATCH NORM
	stage5_out = relu5_2

	upsamp2 = UpSampling3D((3,3,3), name="upsamp2")(stage5_out)
	#upconv2 = Conv3D(128, (2,2,2), padding=padding, name="upconv2")(upsamp2)
	#uprelu2 = LeakyReLU(relu_leak, name="uprelu2")(upconv2)



	stage2_out_cropped = Cropping3D(cropping=30)(stage2_out)
	stage6_in = Concatenate()([upsamp2, stage2_out_cropped])
	filter_count //= filter_scale
	#stage6_in = upsamp2

	conv6_1 = Conv3D(filter_count, (3,3,3), name="conv6_1", padding="valid", kernel_regularizer=reg())(stage6_in)
	relu6_1 = LeakyReLU(relu_leak, name="relu6_1")(conv6_1)
	conv6_2 = Conv3D(filter_count, (3,3,3), name="conv6_2", padding="valid", kernel_regularizer=reg())(relu6_1)
	relu6_2 = LeakyReLU(relu_leak, name="relu6_2")(conv6_2)
	## TODO BATCH NORM
	stage6_out = relu6_2

	upsamp3 = UpSampling3D((3,3,3), name="upsamp3")(stage6_out)
	#upconv3 = Conv3D(64, (2,2,2), padding=padding, name="upconv3")(upsamp3)
	#uprelu3 = LeakyReLU(relu_leak, name="uprelu3")(upconv3)



	stage1_out_cropped = Cropping3D(cropping=102)(stage1_out)
	stage7_in = Concatenate()([upsamp3, stage1_out_cropped])
	filter_count //= filter_scale
	#stage7_in = upsamp3

	conv7_1 = Conv3D(filter_count, (3,3,3), name="conv7_1", padding="valid", kernel_regularizer=reg())(stage7_in)
	relu7_1 = LeakyReLU(relu_leak, name="relu7_1")(conv7_1)
	conv7_2 = Conv3D(filter_count, (3,3,3), name="conv7_2", padding="valid", kernel_regularizer=reg())(relu7_1)
	relu7_2 = LeakyReLU(relu_leak, name="relu7_2")(conv7_2)
	conv7_3 = Conv3D(1, (1,1,1), name="conv7_3", activation="sigmoid", kernel_regularizer=reg())(relu7_2)

	output_layer = conv7_3

	return Model(input_layer, output_layer)




## 156 with factor=2 -> output size of 68
def get_generator_arch_b(init_filters=32, filter_scale=2, relu_leak=0.2, regularization=0.0):

	if regularization > 0.0:
		reg = lambda : L1L2(regularization, regularization)
	else:
		reg = lambda : None

	input_layer = Input(shape = (156,156,156,1))

	filter_count = init_filters
	stage1_in = input_layer
	conv1_1 = Conv3D(filter_count, (3,3,3), name="conv1_1", padding="valid", kernel_regularizer=reg())(stage1_in)
	relu1_1 = LeakyReLU(relu_leak, name="relu1_1")(conv1_1)
	conv1_2 = Conv3D(filter_count, (3,3,3), name="conv1_2", padding="valid", kernel_regularizer=reg())(relu1_1)
	relu1_2 = LeakyReLU(relu_leak, name="relu1_2")(conv1_2)
	## TODO ADD BN
	stage1_out = relu1_2

	pool1 = MaxPooling3D((2,2,2), name="pool1")(stage1_out)


	filter_count *= filter_scale
	stage2_in = pool1

	conv2_1 = Conv3D(filter_count, (3,3,3), name="conv2_1", padding="valid", kernel_regularizer=reg())(stage2_in)
	relu2_1 = LeakyReLU(relu_leak, name="relu2_1")(conv2_1)
	conv2_2 = Conv3D(filter_count, (3,3,3), name="conv2_2", padding="valid", kernel_regularizer=reg())(relu2_1)
	relu2_2 = LeakyReLU(relu_leak, name="relu2_2")(conv2_2)
	## TODO BATCH NORM
	stage2_out = relu2_2


	pool2 = MaxPooling3D((2,2,2), name="pool2")(stage2_out)


	filter_count *= filter_scale
	stage3_in = pool2

	conv3_1 = Conv3D(filter_count, (3,3,3), name="conv3_1", padding="valid", kernel_regularizer=reg())(stage3_in)
	relu3_1 = LeakyReLU(relu_leak, name="relu3_1")(conv3_1)
	conv3_2 = Conv3D(filter_count, (3,3,3), name="conv3_2", padding="valid", kernel_regularizer=reg())(relu3_1)
	relu3_2 = LeakyReLU(relu_leak, name="relu3_2")(conv3_2)
	## TODO BATCH NORM
	stage3_out = relu3_2

	pool3 = MaxPooling3D((2,2,2), name="pool3")(stage3_out)



	filter_count *= filter_scale
	stage4_in = pool3

	conv4_1 = Conv3D(filter_count, (3,3,3), name="conv4_1", padding="valid", kernel_regularizer=reg())(stage4_in)
	relu4_1 = LeakyReLU(relu_leak, name="relu4_1")(conv4_1)
	conv4_2 = Conv3D(filter_count, (3,3,3), name="conv4_2", padding="valid", kernel_regularizer=reg())(relu4_1)
	relu4_2 = LeakyReLU(relu_leak, name="relu4_2")(conv4_2)
	## TODO BATCH NORM
	stage4_out = relu4_2

	upsamp1 = UpSampling3D((2,2,2), name="upsamp1")(stage4_out)
	#upconv1 = Conv3D(64, (2,2,2), padding=padding, name="upconv1")(upsamp1)
	#uprelu1 = LeakyReLU(relu_leak, name="uprelu1")(upconv1)


	stage3_out_cropped = Cropping3D(cropping=4)(stage3_out)
	stage5_in = Concatenate()([upsamp1, stage3_out_cropped])
	filter_count //= filter_scale
	#stage5_in = upsamp1

	conv5_1 = Conv3D(filter_count, (3,3,3), name="conv5_1", padding="valid", kernel_regularizer=reg())(stage5_in)
	relu5_1 = LeakyReLU(relu_leak, name="relu5_1")(conv5_1)
	conv5_2 = Conv3D(filter_count, (3,3,3), name="conv5_2", padding="valid", kernel_regularizer=reg())(relu5_1)
	relu5_2 = LeakyReLU(relu_leak, name="relu5_2")(conv5_2)
	## TODO BATCH NORM
	stage5_out = relu5_2

	upsamp2 = UpSampling3D((2,2,2), name="upsamp2")(stage5_out)
	#upconv2 = Conv3D(128, (2,2,2), padding=padding, name="upconv2")(upsamp2)
	#uprelu2 = LeakyReLU(relu_leak, name="uprelu2")(upconv2)



	stage2_out_cropped = Cropping3D(cropping=16)(stage2_out)
	stage6_in = Concatenate()([upsamp2, stage2_out_cropped])
	filter_count //= filter_scale
	#stage6_in = upsamp2

	conv6_1 = Conv3D(filter_count, (3,3,3), name="conv6_1", padding="valid", kernel_regularizer=reg())(stage6_in)
	relu6_1 = LeakyReLU(relu_leak, name="relu6_1")(conv6_1)
	conv6_2 = Conv3D(filter_count, (3,3,3), name="conv6_2", padding="valid", kernel_regularizer=reg())(relu6_1)
	relu6_2 = LeakyReLU(relu_leak, name="relu6_2")(conv6_2)
	## TODO BATCH NORM
	stage6_out = relu6_2

	upsamp3 = UpSampling3D((2,2,2), name="upsamp3")(stage6_out)
	#upconv3 = Conv3D(64, (2,2,2), padding=padding, name="upconv3")(upsamp3)
	#uprelu3 = LeakyReLU(relu_leak, name="uprelu3")(upconv3)



	stage1_out_cropped = Cropping3D(cropping=40)(stage1_out)
	stage7_in = Concatenate()([upsamp3, stage1_out_cropped])
	filter_count //= filter_scale
	#stage7_in = upsamp3

	conv7_1 = Conv3D(filter_count, (3,3,3), name="conv7_1", padding="valid", kernel_regularizer=reg())(stage7_in)
	relu7_1 = LeakyReLU(relu_leak, name="relu7_1")(conv7_1)
	conv7_2 = Conv3D(filter_count, (3,3,3), name="conv7_2", padding="valid", kernel_regularizer=reg())(relu7_1)
	relu7_2 = LeakyReLU(relu_leak, name="relu7_2")(conv7_2)
	conv7_3 = Conv3D(1, (1,1,1), name="conv7_3", activation="sigmoid", kernel_regularizer=reg())(relu7_2)

	output_layer = conv7_3

	return Model(input_layer, output_layer)





def get_generator(relu_leak=0.2, skip_connections=False, batch_norm=False, regularization=0.0):
	"""Returns valid-padded small generator, mapping 64x64x64 input to 32x32x32 output.
	Architecture is similar to an autoencoder, with skip connections similar to a U-net if enabled.

	Keyword arguments:
	relu_leak -- Alpha parameter to the LeakyReLU layers
	skip_connections -- Determines whether to use skip connections or not
	"""
	if regularization > 0.0:
		reg = lambda : L1L2(regularization, regularization)
	else:
		reg = lambda : None

	input_layer = Input(shape = (64,64,64,1))

	stage1_in = input_layer # feeds into 'stage 1' of the network

	conv1_1 = Conv3D(8, (3,3,3), name="conv1_1", kernel_regularizer=reg())(stage1_in)
	relu1_1 = LeakyReLU(relu_leak, name="relu1_1")(conv1_1)
	conv1_2 = Conv3D(16, (3,3,3), name="conv1_2", kernel_regularizer=reg())(relu1_1)
	relu1_2 = LeakyReLU(relu_leak, name="relu1_2")(conv1_2)

	if batch_norm:
		bn1 = BatchNormalization(momentum=0.8, name="bn1")(relu1_2)
		pool1 = MaxPooling3D((2,2,2), name="pool1")(bn1)
	else:
		pool1 = MaxPooling3D((2,2,2), name="pool1")(relu1_2)


	stage2_in = pool1 # feeds into 'stage 2' of the network

	conv2_1 = Conv3D(32, (3,3,3), name="conv2_1", kernel_regularizer=reg())(stage2_in)
	relu2_1 = LeakyReLU(relu_leak, name="relu2_1")(conv2_1)
	conv2_2 = Conv3D(48, (3,3,3), name="conv2_2", kernel_regularizer=reg())(relu2_1)
	relu2_2 = LeakyReLU(relu_leak, name="relu2_2")(conv2_2)

	if batch_norm:
		bn2 = BatchNormalization(momentum=0.8, name="bn2")(relu2_2)
		pool2 = MaxPooling3D((2,2,2), name="pool2")(bn2)
	else:
		pool2 = MaxPooling3D((2,2,2), name="pool2")(relu2_2)


	stage3_in = pool2 # feeds into 'stage 3' of the network

	conv3 = Conv3D(96, (3,3,3), name="conv3", kernel_regularizer=reg())(stage3_in)
	relu3 = LeakyReLU(relu_leak, name="relu3")(conv3)

	if batch_norm:
		bn3 = BatchNormalization(momentum=0.8, name="bn3")(relu3)
		upsamp1 = UpSampling3D((2,2,2), name="upsamp1")(bn3)
	else:
		upsamp1 = UpSampling3D((2,2,2), name="upsamp1")(relu3)


	stage4_in = upsamp1 # feeds into 'stage 4' of the network

	if skip_connections:
		stage2_out_cropped = Cropping3D(cropping=2)(relu2_2) # (26,)^3 -> (22,)^3
		stage4_in = Concatenate()([stage4_in, stage2_out_cropped])

	conv4_1 = Conv3D(48, (3,3,3), name="conv4_1", kernel_regularizer=reg())(stage4_in)
	relu4_1 = LeakyReLU(relu_leak, name="relu4_1")(conv4_1)
	conv4_2 = Conv3D(32, (3,3,3), name="conv4_2", kernel_regularizer=reg())(relu4_1)
	relu4_2 = LeakyReLU(relu_leak, name="relu4_2")(conv4_2)

	if batch_norm:
		bn4 = BatchNormalization(momentum=0.8, name="bn4")(relu4_2)
		upsamp2 = UpSampling3D((2,2,2), name="upsamp2")(bn4)
	else:
		upsamp2 = UpSampling3D((2,2,2), name="upsamp2")(relu4_2)


	stage5_in = upsamp2 # feeds into 'stage 5' of the network

	if skip_connections:
		stage1_out_cropped = Cropping3D(cropping=12)(relu1_2) # (60,)^3 -> (36,)^3
		stage5_in = Concatenate()([stage5_in, stage1_out_cropped])

	conv5_1 = Conv3D(16, (3,3,3), name="conv5_1", kernel_regularizer=reg())(stage5_in)
	relu5_1 = LeakyReLU(relu_leak, name="relu5_1")(conv5_1)
	conv5_2 = Conv3D(8, (3,3,3), name="conv5_2", kernel_regularizer=reg())(relu5_1)
	relu5_2 = LeakyReLU(relu_leak, name="relu5_2")(conv5_2)

	conv5_3 = Conv3D(1, (1,1,1), activation="sigmoid", name="conv5_3", kernel_regularizer=reg())(relu5_2)
	output_layer = conv5_3

	return Model(input_layer, output_layer)


def get_discriminator(relu_leak=0.2, batch_norm=False, regularization = 0.0):
	"""Returns small discriminator, mapping 32x32x32 into a single output prediction.
	Architecture is fairly standard, two layers of convolutions between max pooling layers.

	Keyword arguments:
	relu_leak -- Alpha parameter to the LeakyReLU layers
	"""

	if regularization > 0.0:
		reg = lambda : L1L2(regularization, regularization)
	else:
		reg = lambda : None

	input_layer = Input(shape = (32,32,32,1))

	conv1_1 = Conv3D(16, (3,3,3), name="conv1_1", kernel_regularizer=reg())(input_layer)
	relu1_1 = LeakyReLU(relu_leak, name="relu1_1")(conv1_1)
	conv1_2 = Conv3D(32, (3,3,3), name="conv1_2", kernel_regularizer=reg())(relu1_1)
	relu1_2 = LeakyReLU(relu_leak, name="relu1_2")(conv1_2)

	if batch_norm:
		bn1 = BatchNormalization(momentum=0.8, name="bn1")(relu1_2)
		pool1 = MaxPooling3D((2,2,2), name="pool1")(bn1)
	else:
		pool1 = MaxPooling3D((2,2,2), name="pool1")(relu1_2)

	conv2_1 = Conv3D(32, (3,3,3), name="conv2_1", kernel_regularizer=reg())(pool1)
	relu2_1 = LeakyReLU(relu_leak, name="relu2_1")(conv2_1)
	conv2_2 = Conv3D(64, (3,3,3), name="conv2_2", kernel_regularizer=reg())(relu2_1)
	relu2_2 = LeakyReLU(relu_leak, name="relu2_2")(conv2_2)


	if batch_norm:
		bn2 = BatchNormalization(momentum=0.8, name="bn2")(relu2_2)
		pool2 = MaxPooling3D((2,2,2), name="pool2")(bn2)
	else:
		pool2 = MaxPooling3D((2,2,2), name="pool2")(relu2_2)

	conv3_1 = Conv3D(64, (3,3,3), name="conv3_1", kernel_regularizer=reg())(pool2)
	relu3_1 = LeakyReLU(relu_leak, name="relu3_1")(conv3_1)
	conv3_2 = Conv3D(96, (1,1,1),  name="conv3_2", kernel_regularizer=reg())(relu3_1)
	relu3_2 = LeakyReLU(relu_leak, name="relu3_2")(conv3_2)

	flat1 = Flatten(name="flat1")(relu3_2)

	dense1 = Dense(32, name="dense1")(flat1)
	relu4 = LeakyReLU(relu_leak, name="relu4")(dense1)

	dense2 = Dense(1, activation="sigmoid", name="dense2")(relu4)
	output_layer = dense2

	return Model(input_layer, output_layer)