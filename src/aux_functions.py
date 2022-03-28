
import numpy as np   


def round_probabilities(x):

	rounded_x = np.copy(x)
	rounded_x[rounded_x >= 0.5] = 1 
	rounded_x[rounded_x < 0.5] = 0

	return rounded_x