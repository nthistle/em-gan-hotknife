from keras.models import load_model
import z5py
import argparse
import numpy as np

def get_argparser():
	parser = argparse.ArgumentParser(description="predict-hotknife")
	parser.add_argument('-g', '--generator', type=str, help="path to generator to use for prediction", required=True)
	parser.add_argument('-c', '--container', type=str, help="path to container to run on", required=True)
	parser.add_argument('-i', '--input', type=str, help="input dataset (within container)", required=True)
	parser.add_argument('-o', '--output', type=str, help="output dataset (within container)", required=True)
	parser.add_argument('-z', '--zgap', type=int, help="location of the gap (z slice)", required=True)
	return parser

def run_prediction(generator_model, input_dataset, output_dataset, zgap):
	input_size = generator_model.input.shape[1].value
	output_size = generator_model.output.shape[1].value
	z_target = zgap - (input_size//2)
	for y in range(0, input_dataset.shape[1] - input_size, output_size):
		for x in range(0, input_dataset.shape[2] - input_size, output_size):
			big_block = input_dataset[z_target:z_target+input_size, y:y+input_size, x:x+input_size]
			big_block = np.expand_dims(np.expand_dims(big_block, 3), 0)
			out_block = generator_model.predict(big_block)
			output_dataset[zgap-(output_size//2):zgap+(output_size//2),
					y+((input_size-output_size)//2):y+((input_size-output_size)//2)+output_size,
					x+((input_size-output_size)//2):x+((input_size-output_size)//2)+output_size] = out_block[0, :, :, :, 0]
			print(x,y)

def main():
	args = get_argparser().parse_args()

	generator = load_model(args.generator)
	container = z5py.File(args.container)
	input_ds = container[args.input]
	output_ds = container[args.output]

	run_prediction(generator, input_ds, output_ds, args.zgap)


if __name__ == "__main__":
	main()