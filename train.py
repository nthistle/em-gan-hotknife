from keras.layers import Input
from keras.losses import mean_squared_error

from data_utils import *

import numpy as np
import tensorflow as tf
import math

import os
import pandas
import argparse


def get_masked_loss(batch_size, output_shape, mask_size, slice_index):

	mask = np.zeros((batch_size,) + output_shape + (1,), dtype=np.float32)
	slices = [slice()]*3

	lower_bound = (output_shape[slice_index] - mask_size)//2
	upper_bound = (output_shape[slice_index] + mask_size)//2

	# set lower to 1
	slices[slice_index] = slice(None,lower_bound)
	mask[:, slices[0], slices[1], slices[2], :] = 1.0

	# set upper to 1
	slices[slice_index] = slice(upper_bound,None)
	mask[:, slices[0], slices[1], slices[2], :] = 1.0

	def masked_loss(y_true, y_pred):
		y_true_masked = tf.multiply(y_true, mask)
		y_pred_masked = tf.multiply(y_pred, mask)
		return mean_squared_error(y_true_masked, y_pred_masked)

	return masked_loss


def apply_noise(samp, std_dev=0.03):
	return np.clip(samp + np.random.normal(0.0, std_dev, size=samp.shape), 0.0, 1.0)


def get_center_of_block(block, target_shape):
	# if block is 64,64,64
	# target is 24
	start_pos = [(a-b)//2 for a, b in zip(block.shape[1:-1], target_shape)]
	slices = [slice(start, start + s) for start, s in zip(start_pos, target_shape)]
	return block[:,*slices]


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, penalty_optimizer,
	epochs, minibatch_size, num_minibatch, instance_noise, instance_noise_profile, input_shape, output_shape,
	generator_mask_size, valid_generator, gap_generator, gap_index, base_save_dir):
    """ Trains the given generator using all the given parameters and generators.

    valid_generator -> should be a generator that returns entirely valid data of size (minibatch_size, *output_shape, 1)
    gap_generator   -> should be a generator that returns data with a gap in the middle of size (minibatch_size, *input_shape, 1)

    """

	discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

	generator.name = "pretrained_generator"
	generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

	penalty_z = Input(shape=input_shape+(1,))
	penalty = Model(penalty_z, generator(penalty_z))
	penalty.compile(loss=get_masked_loss(minibatch_size, output_shape, generator_mask_size, gap_index), optimizer=penalty_optimizer)

	z = Input(shape=input_shape)
	fake_block = generator(z)
	discriminator.trainable = False
	disc_pred = discriminator(fake_block)
	combined = Model(z, disc_pred)

	test_sample = gap_generator.__next__()

	history_cols = ["epoch","d_loss","d_acc","g_loss","g_penalty"]
	history = {col: [] for col in history_cols}

	if not os.path.exists(os.path.join(base_save_dir, "model-saves")):
		os.makedirs(os.path.join(base_save_dir, "model-saves"))

	if not os.path.exists(os.path.join(base_save_dir, "samples")):
		os.makedirs(os.path.join(base_save_dir, "samples"))


	def update_and_print_history(epoch, d_loss, d_acc, g_loss, g_penalty):
		history["epoch"].append(epoch)
		history["d_loss"].append(d_loss)
		history["d_acc"].append(d_acc)
		history["g_loss"].append(g_loss)
		history["g_penalty"].append(g_penalty)
		print(f"Epoch #{epoch+1} [D]: loss: {d_loss[0]} acc: {d_loss[1]}, [G]: loss: {g_loss} penalty: {g_loss_penalty}")


	persistent_sample = np.zeros((18, *input_shape, 1))
	for i in range(math.ceil(18/minibatch_size)):
		samp = gap_generator.__next__()
		persistent_sample[i:min(i+minibatch_size,18)] = samp[:min(minibatch_size,18-i*minibatch_size)]

	def sample_and_write_output(output_directory, epoch, width=32):
		block_height = output_shape[(gap_index+1)%3]
		block_length = output_shape[(gap_index+2)%3]
		slices = [slice()]*3
		im = np.zeros((18*2*block_height, width*block_length))
		#sample_prediction = np.zeros((18, *output_shape, 1))
		for i in range(18): ## TODO: do this in minibatches
			sample_prediction = generator.predict(persistent_sample[i:i+1])[0]
			for j in range(width):
				slices[gap_index] = slice(round(j*output_shape[gap_index]/width),round(j*output_shape[gap_index]/width)+1)
				im[i*2*block_height:(i*2+1)*block_height,block_length*j:block_length*(j+1)] = sample_prediction[slices[0], slices[1], slices[2], 0]
		Image.fromarray(np.clip((255*im).round(),0,255).astype(np.uint8)).save(os.path.join(output_directory, "sample_epoch_%03d.png" % epoch))


	for epoch in range(1,epochs+1):

		d_loss, g_loss, g_loss_penalty = None, None, None

		for _ in range(num_minibatch):
			
			## Train the Discriminator
			gap_data = gap_generator.__next__()
			gen_output = generator.predict(gap_data)

			if instance_noise:
				gen_output = apply_noise(gen_output, instance_noise_profile[epoch])

			valid_data = valid_generator.__next__()

			d_loss_real = discriminator.train_on_batch(valid_data, np.ones((minibatch_size, 1)))
			d_loss_fake = discriminator.train_on_batch(gen_output, np.zeros((minibatch_size, 1)))
			d_loss_new = (1./num_minibatch) * 0.5 * np.add(d_loss_real, d_loss_fake)


			## Train the Generator
			gap_data = gap_generator.__next__()

			## Through the Discriminator
			g_loss_new = (1./num_minibatch) * combined.train_on_batch(gap_data, np.ones((minibatch_size)))

			## Penalty Training
			g_loss_penalty_new = (1./num_minibatch) * penalty.train_on_batch(gap_data, get_center_of_block(gap_data, output_shape))


			## Record Losses
			d_loss = d_loss_new if d_loss is None else np.add(d_loss, d_loss_new)
			g_loss, g_loss_penalty = g_loss_new, g_loss_penalty_new if g_loss is None else np.add(g_loss, g_loss_new), np.add(g_loss_penalty, g_loss_penalty_new)


		update_and_print_history(epoch=epoch, d_loss=d_loss[0], d_acc=d_loss[1], g_loss=g_loss, g_penalty=g_loss_penalty)

		sample_and_write_output(output_directory=os.path.join(base_save_dir, "samples"))

		if (epoch)%15 == 0:
			generator.save(os.path.join(base_save_dir,"model-saves","generator_train_epoch_%03d.h5"%(epoch+1)))
			discriminator.save(os.path.join(base_save_dir,"model-saves","discriminator_train_epoch_%03d.h5"%(epoch+1)))

	with open(os.path.join(base_save_dir,"history.csv"),"w") as f:
		pandas.DataFrame(history).reindex(columns=history_cols).to_csv(f, index=False)

	return generator