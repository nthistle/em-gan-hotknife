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
## Gap Location is the z slice in pixels where the gap is centered
## Gap variance is the +/- amount in z pixels when grabbing gap chunks
## Gap blend determines whether to make the gap_location z slice an average of the layers above and below (+/- 1 z pixel)
## (this is because in one case the gap slice is zeroed out and completely black)
def gap_data_generator_n5(container_path, dataset_path, sample_shape, batch_size, gap_location, gap_variance=1, gap_blend=True):
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
			if gap_blend:
				relative_gap = gap_location - z_start[k]
				batch[k, relative_gap, :, :, 0] = np.mean(
					np.concatenate((batch[k, relative_gap-1:relative_gap, :, :, 0], batch[k, relative_gap+1:relative_gap+2, :, :, 0],),
					 axis=0),
					 axis=0)

		yield batch/255.