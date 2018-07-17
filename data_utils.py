import z5py
import numpy as np

# Uses zyx order
def valid_data_generator_n5(container_path, dataset_path, sample_shape, batch_size):
	data = z5py.File(container_path, use_zarr_format=False)[dataset_path]

	while True:
		batch = np.empty((batch_size, *sample_shape, 1))

		z_start = np.random.randint(0, data.shape[0]-sample_shape[0], batch_size)
		y_start = np.random.randint(0, data.shape[1]-sample_shape[1], batch_size)
		x_start = np.random.randint(0, data.shape[2]-sample_shape[2], batch_size)

		for k in range(batch_size):
			batch[k, :, :, :, 0] = data[z_start[k]:z_start[k]+sample_shape[0],
										y_start[k]:y_start[k]+sample_shape[1],
										x_start[k]:x_start[k]+sample_shape[2]]

		yield batch/255.


# Uses zyx order
def gap_data_generator_n5(container_path, dataset_path, sample_shape, batch_size, gap_location, gap_variance=1):
	data = z5py.File(container_path, use_zarr_format=False)[dataset_path]

	while True:
		batch = np.empty((batch_size, *sample_shape, 1))

		z_start = np.full((batch_size,), gap_location - sample_shape[0]//2)
		z_start += np.random.randint(-gap_variance, gap_variance+1, batch_size)
		y_start = np.random.randint(0, data.shape[1]-sample_shape[1], batch_size)
		x_start = np.random.randint(0, data.shape[2]-sample_shape[2], batch_size)

		for k in range(batch_size):
			batch[k, :, :, :, 0] = data[z_start[k]:z_start[k]+sample_shape[0],
										y_start[k]:y_start[k]+sample_shape[1],
										x_start[k]:x_start[k]+sample_shape[2]]

		yield batch/255.