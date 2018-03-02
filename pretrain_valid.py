from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Reshape, Flatten, Activation, Input, Lambda
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy
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

def write_sampled_output(samp, outp, fname):
	im = np.zeros((640, 640), dtype=np.uint8) # 10 cuts at even spacing, 5 samples, plus 5 outputs
	im[:,:] = 255
	for i in range(5):
		for j in range(10):
			im[128*i:128*i+64,64*j:64*j+64] = (samp[i,j,:,:,0]*255).astype(np.uint8)
			im[128*i+80:128*i+112,64*j+16:64*j+48] = (outp[i,j,:,:,0]*255).astype(np.uint8)
	#resized = imresize(im, 2.0, interp="nearest")
	#Image.fromarray(imresize(im, 2.0, interp="nearest")).save(fname)
	Image.fromarray(im).save(fname)

data_folder = sys.argv[1] #"run_output/"

def main(epochs=200, batch_size=64, num_batches=32, lr=1e-5):

	print("Running pretraining with %d epochs, batch size of %d")
	print("Learning rate is %f" % lr)

	if not os.path.isdir(data_folder):
		os.mkdir(data_folder)

	#generator = get_generator(shape=(32,32,32))
	generator = get_generator_valid(shape=(64,64,64))

	generator.compile(loss='mean_squared_error', optimizer=Adam(lr))

	# for sampling the data for training
	data_gen = h5_nogap_data_generator_valid("hotknifedata.hdf5","volumes/data", (64,64,64), batch_size)

	# just for periodically sampling the generator to see what's going on
	test_gen = h5_nogap_data_generator_valid("hotknifedata.hdf5","volumes/data", (64,64,64), 5)

	train_hist = []

	for epoch in range(epochs):

		g_loss = 0

		for n in range(num_batches):
			samp = data_gen.__next__()
			g_loss += generator.train_on_batch(samp, get_center_of_valid_block(samp))

		g_loss = g_loss/num_batches

		#g_loss = gen_noedge.train_on_batch(latent_samp, latent_samp)

		print("Epoch #%d [G loss: %f]" % (epoch+1, g_loss))

		train_hist.append(g_loss)

		# now save some sample input
		prev = test_gen.__next__()

		outp = generator.predict(prev)

		write_sampled_output(prev, outp, data_folder+"pretrain_epoch_%03d.png"%(epoch+1))

		generator.save(data_folder+"generator_pretrain_epoch_%03d.h5"%(epoch+1))

if __name__ == "__main__":
	main()
