
import pickle
import numpy as np


def save_data(file_name, data):
	pickle.dump(data, open(file_name, 'wb'))


def load_data(file_name):
	return pickle.load(open(file_name, 'rb'))


def round_probabilities(x):

	rounded_x = np.copy(x)
	rounded_x[rounded_x >= 0.5] = 1 
	rounded_x[rounded_x < 0.5] = 0

	return rounded_x