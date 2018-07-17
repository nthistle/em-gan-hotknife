from keras.layers import Conv3D, Conv3DTranspose, UpSampling3D, Dense, Reshape, Flatten, Activation, Input, MaxPooling3D, Cropping3D, Concatenate, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import L1L2

import numpy as np


## 295 with factor=3 -> output size of 83
def get_generator_arch_a(skip_conns=True, init_filters=12, filter_scale=3, relu_leak=0.2, batch_norm=True, bn_momentum=0.8, regularization=0.0):
	init_filters=int(init_filters)
	filter_scale=int(filter_scale)
	relu_leak=float(relu_leak)
	skip_conns=skip_conns in ["True","true",True]
	batch_norm=batch_norm in ["True","true",True]
	bn_momentum=float(bn_momentum)
	regularization=float(regularization)


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

	if batch_norm:
		bn1 = BatchNormalization(momentum=bn_momentum, name="bn1")(relu1_2)
		stage1_out = bn1
	else:
		stage1_out = relu1_2

	pool1 = MaxPooling3D((3,3,3), name="pool1")(stage1_out)



	filter_count *= filter_scale
	stage2_in = pool1

	conv2_1 = Conv3D(filter_count, (3,3,3), name="conv2_1", padding="valid", kernel_regularizer=reg())(stage2_in)
	relu2_1 = LeakyReLU(relu_leak, name="relu2_1")(conv2_1)
	conv2_2 = Conv3D(filter_count, (3,3,3), name="conv2_2", padding="valid", kernel_regularizer=reg())(relu2_1)
	relu2_2 = LeakyReLU(relu_leak, name="relu2_2")(conv2_2)

	if batch_norm:
		bn2 = BatchNormalization(momentum=bn_momentum, name="bn2")(relu2_2)
		stage2_out = bn2
	else:
		stage2_out = relu2_2

	pool2 = MaxPooling3D((3,3,3), name="pool2")(stage2_out)



	filter_count *= filter_scale
	stage3_in = pool2

	conv3_1 = Conv3D(filter_count, (3,3,3), name="conv3_1", padding="valid", kernel_regularizer=reg())(stage3_in)
	relu3_1 = LeakyReLU(relu_leak, name="relu3_1")(conv3_1)
	conv3_2 = Conv3D(filter_count, (3,3,3), name="conv3_2", padding="valid", kernel_regularizer=reg())(relu3_1)
	relu3_2 = LeakyReLU(relu_leak, name="relu3_2")(conv3_2)

	if batch_norm:
		bn3 = BatchNormalization(momentum=bn_momentum, name="bn3")(relu3_2)
		stage3_out = bn3
	else:
		stage3_out = relu3_2

	pool3 = MaxPooling3D((3,3,3), name="pool3")(stage3_out)



	filter_count *= filter_scale
	stage4_in = pool3

	conv4_1 = Conv3D(filter_count, (3,3,3), name="conv4_1", padding="valid", kernel_regularizer=reg())(stage4_in)
	relu4_1 = LeakyReLU(relu_leak, name="relu4_1")(conv4_1)
	conv4_2 = Conv3D(filter_count, (3,3,3), name="conv4_2", padding="valid", kernel_regularizer=reg())(relu4_1)
	relu4_2 = LeakyReLU(relu_leak, name="relu4_2")(conv4_2)

	if batch_norm:
		bn4 = BatchNormalization(momentum=bn_momentum, name="bn4")(relu4_2)
		stage4_out = bn4
	else:
		stage4_out = relu4_2

	upsamp1 = UpSampling3D((3,3,3), name="upsamp1")(stage4_out)
	#upconv1 = Conv3D(64, (2,2,2), padding=padding, name="upconv1")(upsamp1)
	#uprelu1 = LeakyReLU(relu_leak, name="uprelu1")(upconv1)


	if skip_conns:
		stage3_out_cropped = Cropping3D(cropping=6)(stage3_out)
		stage5_in = Concatenate()([upsamp1, stage3_out_cropped])
	else:
		stage5_in = upsamp1
	filter_count //= filter_scale

	conv5_1_name = "conv5_1_" + ("sc" if skip_conns else "nsc")
	conv5_1 = Conv3D(filter_count, (3,3,3), name=conv5_1_name, padding="valid", kernel_regularizer=reg())(stage5_in)
	relu5_1 = LeakyReLU(relu_leak, name="relu5_1")(conv5_1)
	conv5_2 = Conv3D(filter_count, (3,3,3), name="conv5_2", padding="valid", kernel_regularizer=reg())(relu5_1)
	relu5_2 = LeakyReLU(relu_leak, name="relu5_2")(conv5_2)

	if batch_norm:
		bn5 = BatchNormalization(momentum=bn_momentum, name="bn5")(relu5_2)
		stage5_out = bn5
	else:
		stage5_out = relu5_2

	upsamp2 = UpSampling3D((3,3,3), name="upsamp2")(stage5_out)
	#upconv2 = Conv3D(128, (2,2,2), padding=padding, name="upconv2")(upsamp2)
	#uprelu2 = LeakyReLU(relu_leak, name="uprelu2")(upconv2)


	if skip_conns:
		stage2_out_cropped = Cropping3D(cropping=30)(stage2_out)
		stage6_in = Concatenate()([upsamp2, stage2_out_cropped])
	else:
		stage6_in = upsamp2
	filter_count //= filter_scale

	conv6_1_name = "conv6_1_" + ("sc" if skip_conns else "nsc")
	conv6_1 = Conv3D(filter_count, (3,3,3), name=conv6_1_name, padding="valid", kernel_regularizer=reg())(stage6_in)
	relu6_1 = LeakyReLU(relu_leak, name="relu6_1")(conv6_1)
	conv6_2 = Conv3D(filter_count, (3,3,3), name="conv6_2", padding="valid", kernel_regularizer=reg())(relu6_1)
	relu6_2 = LeakyReLU(relu_leak, name="relu6_2")(conv6_2)

	if batch_norm:
		bn6 = BatchNormalization(momentum=bn_momentum, name="bn6")(relu6_2)
		stage6_out = bn6
	else:
		stage6_out = relu6_2

	upsamp3 = UpSampling3D((3,3,3), name="upsamp3")(stage6_out)
	#upconv3 = Conv3D(64, (2,2,2), padding=padding, name="upconv3")(upsamp3)
	#uprelu3 = LeakyReLU(relu_leak, name="uprelu3")(upconv3)


	if skip_conns:
		stage1_out_cropped = Cropping3D(cropping=102)(stage1_out)
		stage7_in = Concatenate()([upsamp3, stage1_out_cropped])
	else:
		stage7_in = upsamp3
	filter_count //= filter_scale

	conv7_1_name = "conv7_1_" + ("sc" if skip_conns else "nsc")
	conv7_1 = Conv3D(filter_count, (3,3,3), name=conv7_1_name, padding="valid", kernel_regularizer=reg())(stage7_in)
	relu7_1 = LeakyReLU(relu_leak, name="relu7_1")(conv7_1)
	conv7_2 = Conv3D(filter_count, (3,3,3), name="conv7_2", padding="valid", kernel_regularizer=reg())(relu7_1)
	relu7_2 = LeakyReLU(relu_leak, name="relu7_2")(conv7_2)
	conv7_3 = Conv3D(1, (1,1,1), name="conv7_3", activation="sigmoid", kernel_regularizer=reg())(relu7_2)

	output_layer = conv7_3

	return Model(input_layer, output_layer)




## 156 with factor=2 -> output size of 68
def get_generator_arch_b(skip_conns=True, init_filters=32, filter_scale=2, relu_leak=0.2, batch_norm=True, bn_momentum=0.8, regularization=0.0):
	init_filters=int(init_filters)
	filter_scale=int(filter_scale)
	relu_leak=float(relu_leak)
	skip_conns=skip_conns in ["True","true",True]
	batch_norm=batch_norm in ["True","true",True]
	bn_momentum=float(bn_momentum)
	regularization=float(regularization)


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

	if batch_norm:
		bn1 = BatchNormalization(momentum=bn_momentum, name="bn1")(relu1_2)
		stage1_out = bn1
	else:
		stage1_out = relu1_2

	pool1 = MaxPooling3D((2,2,2), name="pool1")(stage1_out)



	filter_count *= filter_scale
	stage2_in = pool1

	conv2_1 = Conv3D(filter_count, (3,3,3), name="conv2_1", padding="valid", kernel_regularizer=reg())(stage2_in)
	relu2_1 = LeakyReLU(relu_leak, name="relu2_1")(conv2_1)
	conv2_2 = Conv3D(filter_count, (3,3,3), name="conv2_2", padding="valid", kernel_regularizer=reg())(relu2_1)
	relu2_2 = LeakyReLU(relu_leak, name="relu2_2")(conv2_2)

	if batch_norm:
		bn2 = BatchNormalization(momentum=bn_momentum, name="bn2")(relu2_2)
		stage2_out = bn2
	else:
		stage2_out = relu2_2

	pool2 = MaxPooling3D((2,2,2), name="pool2")(stage2_out)



	filter_count *= filter_scale
	stage3_in = pool2

	conv3_1 = Conv3D(filter_count, (3,3,3), name="conv3_1", padding="valid", kernel_regularizer=reg())(stage3_in)
	relu3_1 = LeakyReLU(relu_leak, name="relu3_1")(conv3_1)
	conv3_2 = Conv3D(filter_count, (3,3,3), name="conv3_2", padding="valid", kernel_regularizer=reg())(relu3_1)
	relu3_2 = LeakyReLU(relu_leak, name="relu3_2")(conv3_2)

	if batch_norm:
		bn3 = BatchNormalization(momentum=bn_momentum, name="bn3")(relu3_2)
		stage3_out = bn3
	else:
		stage3_out = relu3_2

	pool3 = MaxPooling3D((2,2,2), name="pool3")(stage3_out)



	filter_count *= filter_scale
	stage4_in = pool3

	conv4_1 = Conv3D(filter_count, (3,3,3), name="conv4_1", padding="valid", kernel_regularizer=reg())(stage4_in)
	relu4_1 = LeakyReLU(relu_leak, name="relu4_1")(conv4_1)
	conv4_2 = Conv3D(filter_count, (3,3,3), name="conv4_2", padding="valid", kernel_regularizer=reg())(relu4_1)
	relu4_2 = LeakyReLU(relu_leak, name="relu4_2")(conv4_2)

	if batch_norm:
		bn4 = BatchNormalization(momentum=bn_momentum, name="bn4")(relu4_2)
		stage4_out = bn4
	else:
		stage4_out = relu4_2

	upsamp1 = UpSampling3D((2,2,2), name="upsamp1")(stage4_out)
	#upconv1 = Conv3D(64, (2,2,2), padding=padding, name="upconv1")(upsamp1)
	#uprelu1 = LeakyReLU(relu_leak, name="uprelu1")(upconv1)


	if skip_conns:
		stage3_out_cropped = Cropping3D(cropping=4)(stage3_out)
		stage5_in = Concatenate()([upsamp1, stage3_out_cropped])
	else:
		stage5_in = upsamp1
	filter_count //= filter_scale

	conv5_1_name = "conv5_1_" + ("sc" if skip_conns else "nsc")
	conv5_1 = Conv3D(filter_count, (3,3,3), name=conv5_1_name, padding="valid", kernel_regularizer=reg())(stage5_in)
	relu5_1 = LeakyReLU(relu_leak, name="relu5_1")(conv5_1)
	conv5_2 = Conv3D(filter_count, (3,3,3), name="conv5_2", padding="valid", kernel_regularizer=reg())(relu5_1)
	relu5_2 = LeakyReLU(relu_leak, name="relu5_2")(conv5_2)

	if batch_norm:
		bn5 = BatchNormalization(momentum=bn_momentum, name="bn5")(relu5_2)
		stage5_out = bn5
	else:
		stage5_out = relu5_2

	upsamp2 = UpSampling3D((2,2,2), name="upsamp2")(stage5_out)
	#upconv2 = Conv3D(128, (2,2,2), padding=padding, name="upconv2")(upsamp2)
	#uprelu2 = LeakyReLU(relu_leak, name="uprelu2")(upconv2)


	if skip_conns:
		stage2_out_cropped = Cropping3D(cropping=16)(stage2_out)
		stage6_in = Concatenate()([upsamp2, stage2_out_cropped])
	else:
		stage6_in = upsamp2
	filter_count //= filter_scale

	conv6_1_name = "conv6_1_" + ("sc" if skip_conns else "nsc")
	conv6_1 = Conv3D(filter_count, (3,3,3), name=conv6_1_name, padding="valid", kernel_regularizer=reg())(stage6_in)
	relu6_1 = LeakyReLU(relu_leak, name="relu6_1")(conv6_1)
	conv6_2 = Conv3D(filter_count, (3,3,3), name="conv6_2", padding="valid", kernel_regularizer=reg())(relu6_1)
	relu6_2 = LeakyReLU(relu_leak, name="relu6_2")(conv6_2)

	if batch_norm:
		bn6 = BatchNormalization(momentum=bn_momentum, name="bn6")(relu6_2)
		stage6_out = bn6
	else:
		stage6_out = relu6_2

	upsamp3 = UpSampling3D((2,2,2), name="upsamp3")(stage6_out)
	#upconv3 = Conv3D(64, (2,2,2), padding=padding, name="upconv3")(upsamp3)
	#uprelu3 = LeakyReLU(relu_leak, name="uprelu3")(upconv3)


	if skip_conns:
		stage1_out_cropped = Cropping3D(cropping=40)(stage1_out)
		stage7_in = Concatenate()([upsamp3, stage1_out_cropped])
	else:
		stage7_in = upsamp3
	filter_count //= filter_scale

	conv7_1_name = "conv7_1_" + ("sc" if skip_conns else "nsc")
	conv7_1 = Conv3D(filter_count, (3,3,3), name=conv7_1_name, padding="valid", kernel_regularizer=reg())(stage7_in)
	relu7_1 = LeakyReLU(relu_leak, name="relu7_1")(conv7_1)
	conv7_2 = Conv3D(filter_count, (3,3,3), name="conv7_2", padding="valid", kernel_regularizer=reg())(relu7_1)
	relu7_2 = LeakyReLU(relu_leak, name="relu7_2")(conv7_2)
	conv7_3 = Conv3D(1, (1,1,1), name="conv7_3", activation="sigmoid", kernel_regularizer=reg())(relu7_2)

	output_layer = conv7_3

	return Model(input_layer, output_layer)


## TODO ADD NEW DISCRIMINATOR MODELS



def load_weights_compat(new_model, old_model, load_bn=True):
	for layer in old_model.layers:
		is_batchnorm = (layer.name[:2]=="bn")
		is_skipconn = (layer.name[-3:] == "nsc")
		if is_batchnorm:
			if load_bn:
				new_model.get_layer(layer.name).set_weights(layer.get_weights())
		elif is_skipconn:  ## Load into bottom-most of the dims
			new_m_weights = new_model.get_layer(layer.name[:-3] + "sc").get_weights()
			old_m_weights = layer.get_weights()
			new_m_weights[1] = old_m_weights[1] # biases don't change
			new_m_weights[0][:,:,:,:old_m_weights[0].shape[3],:] = old_m_weights[0]
			new_model.get_layer(layer.name[:-3] + "sc").set_weights(new_m_weights)
		else:
			new_model.get_layer(layer.name).set_weights(layer.get_weights())



global ARCHITECTURES

## Architecture Format:
## Key: "architecture name"
## Value: 3-tuple consisting of: (constructor_method, input_shape, output_shape)
## input_shape and output_shape are both integer 3-tuples
ARCHITECTURES = {
	"generator":{
		"a":(get_generator_arch_a,(295,295,295),(83,83,83)),
		"b":(get_generator_arch_b,(156,156,156),(68,68,68))
	},
	"discriminator":{
	}
}

## Set Defaults
ARCHITECTURES["generator"]["default"] = ARCHITECTURES["generator"]["b"]