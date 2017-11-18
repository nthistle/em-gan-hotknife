import numpy as np
from keras.losses import mean_squared_error
from keras import backend as K

def get_mse_masked_loss(mask=None):
	if not mask:
		return mean_squared_error
	def mse_masked_loss(y_true, y_pred):
		return K.mean(K.square(mask * (y_pred - y_true)))
	return mse_masked_loss