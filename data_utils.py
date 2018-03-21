import h5py
import numpy as np

# the gap is around depth 100 (for safety, don't use from 95 to 105)
# so, to get our 30x30x30 block, we're going to pick random depths from 0 to 130 (excl)
# and if it's greater than or eq to 65, we add 40 to it, to safely miss the gap
def h5_nogap_data_generator(data_filename, data_path, sample_shape, batch_size):

	data = h5py.File(data_filename, "r")[data_path]

	while True:
		batch = np.empty((batch_size,) + sample_shape + (1,))

		z_start = np.random.randint(0, 126, batch_size)
		z_start = z_start + (42 * (z_start >= 63))

		x_start = np.random.randint(0, data.shape[1]-sample_shape[1], batch_size)
		y_start = np.random.randint(0, data.shape[2]-sample_shape[2], batch_size)

		for k in range(batch_size):
			data.read_direct(batch,
				np.s_[z_start[k] : z_start[k] + sample_shape[0],
				x_start[k] : x_start[k] + sample_shape[1],
				y_start[k] : y_start[k] + sample_shape[2]],
				np.s_[k, :, :, :, 0])

		yield batch/255.

## Turns (N,64,64,64) into (N,32,32,32) for pretraining/correction
def get_center_of_valid_block(data_block):
	return data_block[:, 16:48, 16:48, 16:48]

def h5_nogap_data_generator_valid(data_filename, data_path, sample_shape, batch_size):

	data = h5py.File(data_filename, "r")[data_path]

	while True:
		batch = np.empty((batch_size,) + sample_shape + (1,))

		z_start = np.random.randint(0, 64, batch_size)
		z_start = z_start + (73 * (z_start >= 32))

		x_start = np.random.randint(0, data.shape[1]-sample_shape[1], batch_size)
		y_start = np.random.randint(0, data.shape[2]-sample_shape[2], batch_size)

		for k in range(batch_size):
			data.read_direct(batch,
				np.s_[z_start[k] : z_start[k] + sample_shape[0],
				x_start[k] : x_start[k] + sample_shape[1],
				y_start[k] : y_start[k] + sample_shape[2]],
				np.s_[k, :, :, :, 0])

		yield batch/255.

# the gap is basically at depth 100, +/- about 1 slice, so the 30x30x30 blocks are
# going to be centered on this in terms of z
def h5_gap_data_generator(data_filename, data_path, sample_shape, batch_size):

	data = h5py.File(data_filename, "r")[data_path]

	while True:
		batch = np.empty((batch_size,) + sample_shape + (1,))

		z_start = np.random.randint(84, 88, batch_size)

		x_start = np.random.randint(0, data.shape[1]-sample_shape[1], batch_size)
		y_start = np.random.randint(0, data.shape[2]-sample_shape[2], batch_size)

		for k in range(batch_size):
			data.read_direct(batch,
				np.s_[z_start[k] : z_start[k] + sample_shape[0],
				x_start[k] : x_start[k] + sample_shape[1],
				y_start[k] : y_start[k] + sample_shape[2]],
				np.s_[k, :, :, :, 0])

		yield batch/255.


# does 64^3 versions
def h5_gap_data_generator_valid(data_filename, data_path, sample_shape, batch_size):

	data = h5py.File(data_filename, "r")[data_path]

	while True:
		batch = np.empty((batch_size,) + sample_shape + (1,))

		z_start = np.random.randint(66, 71, batch_size)

		x_start = np.random.randint(0, data.shape[1]-sample_shape[1], batch_size)
		y_start = np.random.randint(0, data.shape[2]-sample_shape[2], batch_size)

		for k in range(batch_size):
			data.read_direct(batch,
				np.s_[z_start[k] : z_start[k] + sample_shape[0],
				x_start[k] : x_start[k] + sample_shape[1],
				y_start[k] : y_start[k] + sample_shape[2]],
				np.s_[k, :, :, :, 0])

		yield batch/255.



def write_sampled_output(samp, outp, fname, width=16):
	im = np.zeros((640, 64*width), dtype=np.uint8) # cuts at even spacing, 5 samples, plus 5 outputs
	im[:,:] = 255
	for i in range(5):
		for j in range(width):
			im[128*i:128*i+64,64*j:64*j+64] = (samp[i,round(j*64./width),:,:,0]*255).astype(np.uint8)
			im[128*i+80:128*i+112,64*j+16:64*j+48] = (outp[i,round(j*32./width),:,:,0]*255).astype(np.uint8)
	Image.fromarray(im).save(fname)


## Same as the other one, just visually slightly different
## (only takes the center piece of samp)
def write_sampled_output_even(samp, outp, fname, width=16):
	im = np.zeros((320, 32*width), dtype=np.uint8) # cuts at even spacing, 5 samples, plus 5 outputs
	samp = get_center_of_valid_block(samp)
	im[:,:] = 255
	for i in range(5):
		for j in range(width):
			im[64*i:64*i+32,32*j:32*j+32] = (samp[i,round(j*32./width),:,:,0]*255).astype(np.uint8)
			im[64*i+32:64*i+64,32*j+32:32*j+64] = (outp[i,round(j*32./width),:,:,0]*255).astype(np.uint8)
	Image.fromarray(im).save(fname)