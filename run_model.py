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
	generator = architecture_specs[0](**generator_constructor_args) #invoke constructor with provided args

	generator_optimizer = Adam(lr=float(pretrain_args["generator_learning_rate"]))
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



def handle_train(generator, global_args, train_args):
	""" Handles the main training of the model. Does the busywork/cleaning up needed to call train.train().

	generator - Pretrained model, if pretraining was also done this run. Otherwise, None, in which case
				it is loaded from train_args['pretrained_model']
	"""

	pass


def main():
	args = get_argparser().parse_args()
	config = configparser.ConfigParser()
	config.read(args.config)
	train = str2bool(config["global"]["train"], "global.train")
	pretrain = str2bool(config["global"]["pretrain"], "global.pretrain")

	if not train and not pretrain:
		raise Exception("Nothing to do!")

	generator = None

	if pretrain:
		# handle pretrain here
		generator = handle_pretrain(config["global"], config["pretrain"])


	if train:
		# handle train here
		handle_train(generator, config["global"], config["pretrain"])


if __name__=="__main__":
	main()