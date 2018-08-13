from keras.layers import Input
from keras.losses import mean_squared_error
from PIL import Image

import numpy as np
import tensorflow as tf
import math

import os
import pandas
import argparse


def get_center_of_block(block, target_shape):
	start_pos = [(a-b)//2 for a, b in zip(block.shape[1:-1], target_shape)]
	slices = [slice(start, start + s) for start, s in zip(start_pos, target_shape)]
	return block[:,slices[0],slices[1],slices[2]]


## Minibatch size usually has to be 2 or fewer for these models, otherwise it will run out of memory
## on the GPU when trying to allocate. For the largest model (a), this has to be 1.

def pretrain(generator, generator_optimizer, epochs, minibatch_size, num_minibatch, input_shape, output_shape,
	valid_generator, base_save_dir):
	""" Pre-trains the given generator using the given parameters and valid data generator.

	valid_generator -> should be a generator that returns entirely valid data of size (minibatch_size, *output_shape, 1)

	"""

	generator.compile(loss=mean_squared_error, optimizer=generator_optimizer)

	history_cols = ["epoch","loss"]
	history = {col: [] for col in history_cols}

	# Not worthwhile to save generator checkpoints
	if not os.path.exists(os.path.join(base_save_dir, "samples")):
		os.makedirs(os.path.join(base_save_dir, "samples"))

	def update_and_print_history(epoch, loss):
		history["epoch"].append(epoch)
		history["loss"].append(loss)
		print(f"Epoch #{epoch} [G]: loss: {g_loss}")

	persistent_sample = np.zeros((18, *input_shape, 1))
	for i in range(math.ceil(18/minibatch_size)):
		samp = valid_generator.__next__()
		persistent_sample[i*minibatch_size:min(i*minibatch_size+minibatch_size,18)] = samp[:min(minibatch_size,18-i*minibatch_size)]
	persistent_sample_center = get_center_of_block(persistent_sample, output_shape)

	def sample_and_write_output(output_directory, epoch, width=32):
		block_height = output_shape[1]
		block_length = output_shape[2]
		slices = [slice(None)]*3
		im = np.zeros((18*2*block_height, width*block_length))
		for i in range(18): ## TODO: do this in minibatches
			sample_prediction = generator.predict(persistent_sample[i:i+1])[0]
			for j in range(width):
				slices[0] = slice(round(j*output_shape[0]/width),round(j*output_shape[0]/width)+1)
				im[i*2*block_height:(i*2+1)*block_height,block_length*j:block_length*(j+1)] = persistent_sample_center[i, slices[0], slices[1], slices[2], 0]
				im[(i*2+1)*block_height:(i*2+2)*block_height,block_length*j:block_length*(j+1)] = sample_prediction[slices[0], slices[1], slices[2], 0]
		Image.fromarray(np.clip((255*im).round(),0,255).astype(np.uint8)).save(os.path.join(output_directory, "sample_epoch_%03d.png" % epoch))


	for epoch in range(1,epochs+1):

		g_loss = None

		for _ in range(num_minibatch):

			## Train the Generator
			valid_data = valid_generator.__next__()

			g_loss_new = (1./num_minibatch) * generator.train_on_batch(valid_data, get_center_of_block(valid_data, output_shape))

			## Record Loss
			g_loss = g_loss_new if g_loss is None else np.add(g_loss, g_loss_new)


		update_and_print_history(epoch=epoch, loss=g_loss)

		sample_and_write_output(output_directory=os.path.join(base_save_dir, "samples"), epoch=epoch)


	with open(os.path.join(base_save_dir,"history.csv"),"w") as f:
		pandas.DataFrame(history).reindex(columns=history_cols).to_csv(f, index=False)

	return generator