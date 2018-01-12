from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras_adversarial import AdversarialModel, simple_gan, AdversarialOptimizerSimultaneous
import numpy as np

from PIL import Image
from scipy.misc import imresize

from util import *

## Basically, this will have some artificial GAN training task, and verify that this can be learned
## relatively easily (and reliably?), and then will use the new optimizer (with extra optimizer/loss)
## to demonstrate that this modifies the output in some way

## Test 1

## Going to have the generator receive an image of random noise as input, except with the area near
## the sides of the image whited out

## The discriminator will be trying to tell apart the generator output from real random noise

## Ideally, without the extra punishment, the generator will just produce random noise that in not
## similar to the input noise in the middle of the image.
## Then, I will add in a "punishment" for deviating from the middle 4x4, and hopefully then it will
## keep the middle 4x4 similar.

def generator_latent_gen(shape=(10,10),batch_size=1):
	rng = random_noise_gen(shape, batch_size)
	for n in rng:
		n[:,:2,:,0] = 0.0
		n[:,-2:,:,0] = 0.0
		n[:,:,:2,0] = 0.0
		n[:,:,-2:,0] = 0.0
		yield n

def full_data_generator(shape=(10,10),targets=[1,0,0,1],batch_size=16):
	dat_x = random_noise_gen(shape, batch_size)
	dat_y = [np.ones((batch_size, 1)) if i==1 else np.zeros((batch_size, 1)) for i in targets]

	for x in dat_x:
		yield (x, dat_y)

def random_noise_gen(shape=(10,10), batch_size=1):
	while True:
		yield np.random.random((batch_size,) + shape + (1,))

def display_as_img(arr,scalefactor=255,resize=None):
	if resize is None:
		Image.fromarray(np.clip(scalefactor*arr,0,255).astype(np.uint8)).show()
	else:
		Image.fromarray(np.clip(imresize(scalefactor*arr, resize, interp="nearest"),0,255).astype(np.uint8)).show()

def lambda_generator_latent(x, shape=(10,10,1)):
	#zeros_mod = lambda x,s : K.fill(K.stack([K.shape(x)[0]] + [s]), 0.0)
	#zeros_mod = lambda x,s : K.zeros_like(K.stack([K.shape(x)[0]] + [i for i in s]), dtype="float32")
	zeros_mod = lambda x,s : K.zeros_like(K.placeholder(shape=((x.shape[0],)+s), dtype="float32"))

	edge_size = 2
	shape_no_edges = (shape[0]-2*edge_size,shape[1]-2*edge_size,1)
	shape_edges1 = (edge_size,shape[1]-2*edge_size,1)
	shape_edges2 = (shape[0],edge_size,1)
	samples_shape = (K.shape(x)[0],)
	#y = K.random_uniform(samples_shape+shape_no_edges)
	#y = K.concatenate([zeros_mod(samples_shape+shape_edges1),y,zeros_mod(samples_shape+shape_edges1)],axis=1)
	#y = K.concatenate([zeros_mod(samples_shape+shape_edges2),y,zeros_mod(samples_shape+shape_edges2)],axis=2)
	y = K.random_uniform(samples_shape+shape_no_edges)
	y = K.concatenate([zeros_mod(x,shape_edges1),y,zeros_mod(x,shape_edges1)],axis=1)
	y = K.concatenate([zeros_mod(x,shape_edges2),y,zeros_mod(x,shape_edges2)],axis=2)
	return y

def custom_latent_sampling(latent_shape=(10,10,1)):
	return Lambda(lambda x: lambda_generator_latent(x, latent_shape),
		output_shape=lambda x: ((x[0],) + latent_shape))

def run_test_1_nop():
	generator = Sequential()

	generator.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(10,10,1)))
	generator.add(Conv2D(8, (3,3), padding="same", activation="relu"))
	generator.add(Conv2D(4, (3,3), padding="same", activation="relu"))
	generator.add(Conv2D(4, (3,3), padding="same", activation="relu"))
	generator.add(Conv2D(1, (3,3), padding="same", activation="relu"))

	generator.summary()

	discriminator = Sequential()

	discriminator.add(Conv2D(16, (3,3), activation="relu", input_shape=(10,10,1)))
	discriminator.add(Conv2D(8, (3,3), activation="relu"))
	discriminator.add(Reshape((288,)))
	discriminator.add(Dense(8, activation="relu"))
	discriminator.add(Dense(1, activation="sigmoid"))

	discriminator.summary()

	gan = simple_gan(generator, discriminator, custom_latent_sampling())

	model = AdversarialModel(base_model=gan,
		player_params=[generator.trainable_weights, discriminator.trainable_weights],
		player_names=["generator","discriminator"])

	model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
		player_optimizers=[Adam(1e-4), Adam(1e-3)],
		loss='binary_crossentropy')

	#history = model.fit_generator(random_noise_gen(batch_size=16), 100, epochs=3)

	history = model.fit_generator(full_data_generator((10,10), [1,0,0,1], 16), 100, epochs=3)

	return generator, discriminator, history


## Test the masked mean squared error loss

# def test_masked_mse_loss(shape=(30,30,30)):
# 	return train_model(get_model(shape), shape)

# def get_model(shape=(30,30,30), mask=None):
# 	if mask is None:
# 		mask = get_standard_mask(shape=shape, cut_width=20)

# 	model = Sequential()

# 	model.add(Conv3D(16, (3,3,3), padding="same", input_shape=(shape+(2,))))
# 	model.add(LeakyReLU(0.2))

# 	model.add(Conv3D(8, (3,3,3), padding="same"))
# 	model.add(LeakyReLU(0.2))

# 	model.add(Conv3D(1, (1,1,1), padding="same", input_shape=(shape+(1,)), activation="relu"))
# 	###model.add(Conv3D(1, (1,1,1), padding="same", activation="relu"))

# 	model.summary()

# 	loss_func = get_mse_masked_loss(mask)

# 	model.compile(Adam(), loss=loss_func, metrics=["mae"])
	
# 	return model


# def train_model(model, shape=(30,30,30)):

# 	#model.add(Conv3D(16, (3,3,3), padding="same", input_shape=(shape+(1,))))
# 	#model.add(LeakyReLU(0.2))

# 	#model.add(Conv3D(16, (3,3,3), padding="same"))
# 	#model.add(LeakyReLU(0.2))

# 	#model.add(Conv3D(8, (3,3,3), padding="same"))
# 	#model.add(LeakyReLU(0.2))

# 	model.fit_generator(random_noise_gen(shape=shape), 100, 2)

# 	return model