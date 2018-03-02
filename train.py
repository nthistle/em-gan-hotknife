from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras import backend as K
from PIL import Image
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
import os
from util import *
from discriminator import *
from generator import *
from data_utils import *
import sys

def get_masked_loss(batch_size):
	mask = np.zeros((batch_size, 32, 32, 32, 1), dtype=np.float32)
	mask[:,:8,:,:,:] = 1.0
	mask[:,8,:,:,:] = 0.7
	mask[:,9,:,:,:] = 0.3
	mask[:,-8:,:,:,:] = 1.0
	mask[:,-9,:,:,:] = 0.7
	mask[:,-10,:,:,:] = 0.3
	def masked_loss(y_true, y_pred):
		y_true_masked = tf.multiply(y_true, mask)
		y_pred_masked = tf.multiply(y_pred, mask)
		return mean_squared_error(y_true_masked, y_pred_masked)
	return masked_loss

def write_sampled_output(samp, outp, fname, width=16):
	im = np.zeros((320, 32*width), dtype=np.uint8) # cuts at even spacing, 5 samples, plus 5 outputs
	for i in range(5):
		for j in range(width):
			im[64*i:64*i+32,32*j:32*(j+1)] = (samp[i,round(j*32./width),:,:,0]*255).astype(np.uint8)
			im[64*i+32:64*i+64,32*j:32*(j+1)] = (outp[i,round(j*32./width),:,:,0]*255).astype(np.uint8)
	Image.fromarray(imresize(im, 2.0, interp="nearest")).save(fname)

data_folder = sys.argv[1]#"run_output/"

#should be 32
def main(epochs=25, batch_size=64, num_batches=32, disc_lr=1e-7, gen_lr=1e-6):

	print("Running training with %d epochs, batch size of %d")
	print("Learning rates are D:%f, G:%f" % (disc_lr,gen_lr))

	if not os.path.isdir(data_folder):
		os.mkdir(data_folder)

	discriminator = get_discriminator(shape=(32,32,32))
	discriminator.compile(loss='binary_crossentropy', optimizer=Adam(disc_lr), metrics=['accuracy'])

	generator = load_model("generator_pretrain_epoch_200.h5") #get_generator(shape=(32,32,32))
	generator.name = "model_pre"
	generator.compile(loss=get_masked_loss(batch_size), optimizer=Adam(gen_lr))

	z = Input(shape=(32,32,32,1))
	img = generator(z)

	discriminator.trainable = False

	valid = discriminator(img)

	combined = Model(z, valid)
	combined.compile(loss='binary_crossentropy', optimizer=Adam(gen_lr))

	# for sampling the data for training
	fake_gen = h5_gap_data_generator("hotknifedata.hdf5","volumes/data", (32,32,32), batch_size)

	# for training discriminator on what is real
	real_gen = h5_nogap_data_generator("hotknifedata.hdf5","volumes/data", (32,32,32), batch_size)

	# just for periodically sampling the generator to see what's going on
	test_gen = h5_gap_data_generator("hotknifedata.hdf5","volumes/data", (32,32,32), 5)

	for epoch in range(epochs):

		g_loss = None
		g_loss_penalty = None
		d_loss = None

		for n in range(num_batches): # do n minibatches

			# train discriminator
			latent_samp = fake_gen.__next__() # input to generator
			gen_output = generator.predict(latent_samp)

			real_data = real_gen.__next__()

			d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
			d_loss_fake = discriminator.train_on_batch(gen_output, np.zeros((batch_size, 1)))
			d_loss_new = (1./num_batches) * 0.5 * np.add(d_loss_real, d_loss_fake)

			if d_loss is None:
				d_loss = d_loss_new
			else:
				d_loss = np.add(d_loss, d_loss_new)

			# train generator
			latent_samp = fake_gen.__next__()

			g_loss_new = (1./num_batches) * combined.train_on_batch(latent_samp, np.ones((batch_size, 1)))

			g_loss_penalty_new = (1./num_batches) * generator.train_on_batch(latent_samp, latent_samp)

			if g_loss is None:
				g_loss = g_loss_new
				g_loss_penalty = g_loss_penalty_new
			else:
				g_loss = np.add(g_loss, g_loss_new)
				g_loss_penalty = np.add(g_loss_penalty, g_loss_penalty_new)

		print("Epoch #%d [D loss: %f acc: %f] [G loss: %f penalty: %f]" % (epoch+1, d_loss[0], d_loss[1], g_loss, g_loss_penalty))

		# now save some sample input
		prev = test_gen.__next__()

		outp = generator.predict(prev)

		write_sampled_output(prev, outp, data_folder+"train_epoch_%03d.png"%(epoch+1))

		generator.save(data_folder+"train_epoch_%03d.h5"%(epoch+1))

if __name__ == "__main__":
	main()
