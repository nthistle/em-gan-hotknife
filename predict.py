import h5py
import numpy as np

from PIL import Image
from keras.models import load_model

import numpy as np
import os
import argparse


def main(generator_filename, datafile, z_level, output):
	generator = load_model(generator_filename)

	data = h5py.File(datafile, "r")["volumes/data"]

	predicted = np.zeros((32, data.shape[1], data.shape[2]), dtype=data.dtype)

	z_start = z_level - 32

	for x in range(0, data.shape[1]-64, 32):
		for y in range(0, data.shape[2]-64, 32):
			block = data[z_level:z_level+64,x:x+64,y:y+64]/255.
			pred = generator.predict(np.array([block]))
			predicted[z_level+16:z_level+48,x+16:x+48,y+16:y+48] = (255*pred[0]).astype(data.dtype)

	for z in range(32):
		Image.fromarray(predicted[z]).save("%02d.tiff" % z)


def generate_argparser():
	parser = argparse.ArgumentParser(description="Predict on a volume of data")
	parser.add_argument('-g','--generator', type=str, help="generator model (h5) to predict with", required=True)
	parser.add_argument('-df','--datafile', type=str, help="data file (hdf5) to predict to", required=True)
	parser.add_argument('-z','--z_level', type=int, help="z level that the gap is located at (center)", required=True)
	#parser.add_argument('-o','--output', type=str, help="output data file (hdf5) to write predictions to", required=True)
	parser.add_argument('-o','--output', type=str, help="output file directory to write tiff stack to", required=True)
	return parser

if __name__ == "__main__":
	args = generate_argparser().parse_args()
	main(args.generator,
		args.datafile,
		args.z_level,
		args.output)