import numpy as np
from keras.losses import mean_squared_error
from keras import backend as K

def get_mse_masked_loss(mask=None):
	if not mask:
		return mean_squared_error
	def mse_masked_loss(y_true, y_pred):
		return K.mean(K.square(mask * (y_pred - y_true)))
	return mse_masked_loss

def get_standard_mask(shape=(50,50,50), cut_width=8, cut_axis=0, fade_size=None):
	mask = np.ones(shape=shape)
	cut_start = (shape[0]-cut_width)//2

	mask[cut_start:cut_start+cut_width,:,:] = 0.0 # temporarily hard coded

	if not fade_size:
		fade_size = (shape[0]-cut_width)//4

	for i in range(fade_size):
		mask[cut_start-i-1,:,:] = i/fade_size
		mask[cut_start+cut_width+i,:,:] = i/fade_size

	return mask