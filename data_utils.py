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
