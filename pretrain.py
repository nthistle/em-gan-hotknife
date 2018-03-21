from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from keras.regularizers import L1L2

from keras import backend as K

from PIL import Image
from scipy.misc import imresize

from models import *
from data_utils import *

import numpy as np
import tensorflow as tf

import os
import sys
import argparse


def get_masked_loss(batch_size):

	mask = np.zeros((batch_size, 32, 32, 32, 1), dtype=np.float32)
	mask[:,:7,:,:,:] = 1.0
	mask[:,7,:,:,:] = 0.7
	mask[:,8,:,:,:] = 0.3
	mask[:,-7:,:,:,:] = 1.0
	mask[:,-8,:,:,:] = 0.7
	mask[:,-9,:,:,:] = 0.3

	def masked_loss(y_true, y_pred):
		y_true_masked = tf.multiply(y_true, mask)
		y_pred_masked = tf.multiply(y_pred, mask)
		return mean_squared_error(y_true_masked, y_pred_masked)

	return masked_loss



def main(epochs=25, batch_size=64, num_batches=32,
	gen_lr=1e-6, data_file="hotknifedata.hdf5", output_folder="run_output"):

	print("Running em-hotknife GAN pre-training for %d epochs with parameters:" % epochs)
	print("Generator LR: %f", % gen_lr)

	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	generator = get_generator()
	generator.compile(loss='mean_squared_error', optimizer=Adam(gen_lr))

	# for pretraining generator
	real_gen = h5_nogap_data_generator_valid(data_file, "volumes/data", (64,64,64), batch_size)

	# for sampling pretrained generator
	test_gen = h5_nogap_data_generator_valid(data_file, "volumes/data", (64,64,64), 5)


	## Just do an "Epoch 0" test
	latent_samp = fake_gen.__next__()
	gen_output = generator.predict(latent_samp)
	real_data = real_gen.__next__()
	g_loss = generator.test_on_batch(latent_samp, get_center_of_valid_block(latent_samp))
	print("Epoch 0 [D loss: %f acc: %f] [G loss: %f penalty: %f]" % (d_loss[0], d_loss[1], g_loss, g_loss_penalty))
	print("="*50)
	## End our Epoch 0 stuff

	for epoch in range(epochs):

		g_loss = None

		for n in range(num_batches): # do n minibatches

			# train generator
			pret_samp = real_gen.__next__()

			## Now penalty instead of generator
			g_loss_new = (1./num_batches) * penalty.train_on_batch(pret_samp, get_center_of_valid_block(pret_samp))

			if g_loss is None:
				g_loss = g_loss_new
			else:
				g_loss = np.add(g_loss, g_loss_new)

		print("Epoch #%d [G loss: %f]" % (epoch+1, g_loss))

		# now save some sample input
		prev = test_gen.__next__()

		outp = generator.predict(prev)

		write_sampled_output_even(prev, outp, os.path.join(output_folder,"pretrain_epoch_%03d.png"%(epoch+1)))

		if (epoch+1)%5 == 0:
			generator.save(os.path.join(output_folder,"generator_pretrain_epoch_%03d.h5"%(epoch+1)))

	generator.save(os.path.join(output_folder,"generator_pretrain_final.h5"))


def generate_argparser():
	parser = argparse.ArgumentParser(description="Train em-hotknife GAN")
	parser.add_argument('-df','--datafile', type=str, help="data file (hdf5) to sample from", required=True)
	parser.add_argument('-ne','--epochs', type=int, help="number of epochs to train for", default=50)
	parser.add_argument('-glr','--gen_lr', type=float, help="generator learning rate", required=True)
	parser.add_argument('-o','--output', type=str, help="folder/directory to output data to", required=True)
	return parser

if __name__ == "__main__":
	args = generate_argparser().parse_args()
	main(args.generator,
		epochs = args.epochs,
		gen_lr = args.gen_lr,
		output_folder = args.output,
		data_file = args.datafile
		)
