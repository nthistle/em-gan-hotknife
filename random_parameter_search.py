import argparse
import configparser
from random import uniform, randint, random

def get_argparser():
	parser = argparse.ArgumentParser(description="random-parameter-search")
	parser.add_argument('-c','--config', type=str, help="Base config file to use for making random parameter config files", required=True)
	parser.add_argument('-o','--output', type=str, help="Output config file to write random parameter config to", required=True)
	return parser

def main():
	args = get_argparser().parse_args()
	config = configparser.ConfigParser()
	config.read(args.config)

	d_lr = uniform(-8,-4)
	g_lr = uniform(-8,-4)
	p_lr = uniform(-8,-4)
	while d_lr > g_lr + 0.75 or g_lr > d_lr + 2: # don't want too imbalanced random params
		d_lr = uniform(-8,-4)
		g_lr = uniform(-8,-4)

	config["train"]["discriminator_learning_rate"] = str(10**d_lr)
	config["train"]["generator_learning_rate"] = str(10**g_lr)
	config["train"]["penalty_learning_rate"] = str(10**p_lr)

	config["train"]["generator_mask_size"] = str(randint(8,20))
	config["train"]["discriminator_arg_dropout"] = "0.0" if random()<0.4 else str(uniform(0, 2))
	config["train"]["discriminator_arg_bn_momentum"] = str(uniform(0.8,0.999))

	with open(args.output, "w") as output_cfg:
		config.write(output_cfg)

if __name__=="__main__":
	main()