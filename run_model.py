#!/usr/bin/env python3
from keras.layers import Input
from keras.models import load_model
from keras.losses import mean_squared_error
from keras.optimizers import Adam, SGD
from keras.regularizers import L1L2

from keras import backend as K

import models
import train
import pretrain
import data_utils

import numpy as np
import tensorflow as tf

import os
import pandas
import sys
import shutil

import argparse
import configparser

def get_argparser():
	parser = argparse.ArgumentParser(description="em-gan-hotknife")
	parser.add_argument('-c','--config', type=str, help="path to a .cfg or .ini with run parameters and specifications", required=True)
	return parser

def str2bool(v, var=None):
	if v.lower() in ('true', 't', 'yes', 'y', '1'):
		return True
	elif v.lower() in ('false', 'f', 'no', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected%s!' % ('for variable %s' % var if var else ''))

def handle_pretrain(global_args, pretrain_args):
	architecture_specs = models.ARCHITECTURES["generator"][pretrain_args["generator_architecture"]]

	generator_constructor_args = {arg[14:]:pretrain_args[arg] for arg in pretrain_args if arg[:14]=="generator_arg_"}
	print(f"Detected {len(generator_constructor_args)} arguments for the generator constructor")
	generator = architecture_specs[0](**generator_constructor_args) #invoke constructor with provided args

	generator_optimizer = Adam(lr=float(pretrain_args["generator_learning_rate"])) ## TODO do this based on cfg
	num_epochs = int(pretrain_args["num_epochs"])
	num_minibatch = int(pretrain_args["num_minibatch"])
	minibatch_size = int(pretrain_args["minibatch_size"])
	input_shape = architecture_specs[1]
	output_shape = architecture_specs[2]

	base_save_dir = os.path.join(global_args["run_output"],"pretrain")
	os.makedirs(base_save_dir)

	valid_data_generator = data_utils.valid_data_generator_n5(global_args["valid_container"], global_args["valid_dataset"], input_shape, minibatch_size)

	pretrain.pretrain(generator=generator, generator_optimizer=generator_optimizer, epochs=num_epochs,
		minibatch_size=minibatch_size, num_minibatch=num_minibatch, input_shape=input_shape, output_shape=output_shape,
		valid_generator=valid_data_generator, base_save_dir=base_save_dir)

	if models.autodetect_skipconn(generator):
		# if it has skip connections
		generator.save(os.path.join(base_save_dir, "pretrained-generator.h5"))
	else:
		# if it doesn't, add them
		generator.save(os.path.join(base_save_dir, "pretrained-generator-nsc.h5"))
		generator_constructor_args["skip_conns"] = True
		generator_sc = architecture_specs[0](**generator_constructor_args)
		models.load_weights_compat(generator_sc, generator, True)
		generator_sc.save(os.path.join(base_save_dir, "pretrained-generator.h5"))

	return generator



def handle_train(generator, global_args, train_args):
	""" Handles the main training of the model. Does the busywork/cleaning up needed to call train.train().

	generator - Pretrained model, if pretraining was also done this run. Otherwise, None, in which case
				it is loaded from train_args['pretrained_model']
	"""
	d_architecture_specs = models.ARCHITECTURES["discriminator"][train_args["discriminator_architecture"]]

	discriminator_constructor_args = {arg[18:]:train_args[arg] for arg in train_args if arg[:18]=="discriminator_arg_"}
	print(f"Detected {len(discriminator_constructor_args)} arguments for the discriminator constructor")
	discriminator = d_architecture_specs[0](**discriminator_constructor_args)

	if generator is None:
		generator = load_model(train_args["pretrained_model"])

	generator_optimizer = Adam(lr=float(train_args["generator_learning_rate"]))
	discriminator_optimizer = Adam(lr=float(train_args["discriminator_learning_rate"]))
	penalty_optimizer = Adam(lr=float(train_args["penalty_learning_rate"]))
	num_epochs = int(train_args["num_epochs"])
	num_minibatch = int(train_args["num_minibatch"])
	minibatch_size = int(train_args["minibatch_size"])

	instance_noise = str2bool(train_args["instance_noise"],"train.instance_noise")
	instance_noise_profile = [float(train_args["instance_noise_std_dev"])]*num_epochs

	input_shape = *(dim.value for dim in generator.input.shape[1:4]),
	output_shape = *(dim.value for dim in generator.output.shape[1:4]),

	generator_mask_size = int(train_args["generator_mask_size"])

	gap_index = int(global_args["gap_location"])

	valid_generator = data_utils.valid_data_generator_n5(global_args["valid_container"], global_args["valid_dataset"], output_shape, minibatch_size)
	gap_generator = data_utils.gap_data_generator_n5(global_args["gap_container"], global_args["gap_dataset"], input_shape, minibatch_size, gap_index)

	base_save_dir = os.path.join(global_args["run_output"], "train")
	os.makedirs(base_save_dir)

	train.train(generator=generator, discriminator=discriminator, generator_optimizer=generator_optimizer,
		discriminator_optimizer=discriminator_optimizer, penalty_optimizer=penalty_optimizer, epochs=num_epochs,
		minibatch_size=minibatch_size, num_minibatch=num_minibatch, instance_noise=instance_noise,
		instance_noise_profile=instance_noise_profile, input_shape=input_shape, output_shape=output_shape,
		generator_mask_size=generator_mask_size, valid_generator=valid_generator, gap_generator=gap_generator,
		gap_index=0, base_save_dir=base_save_dir)

	generator.save(os.path.join(base_save_dir, "generator-final.h5"))
	discriminator.save(os.path.join(base_save_dir, "discriminator-final.h5"))

	return (generator, discriminator)




def main():
	args = get_argparser().parse_args()
	config = configparser.ConfigParser()
	config.read(args.config)
	train = str2bool(config["global"]["train"], "global.train")
	pretrain = str2bool(config["global"]["pretrain"], "global.pretrain")

	if not train and not pretrain:
		raise Exception("Nothing to do!")

	save_path = config["global"]["run_output"]

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	shutil.copyfile(args.config, os.path.join(save_path,"run_parameters.cfg"))

	generator = None

	if pretrain:
		# handle pretrain here
		generator = handle_pretrain(config["global"], config["pretrain"])


	if train:
		# handle train here
		handle_train(generator, config["global"], config["train"])


if __name__=="__main__":
	main()