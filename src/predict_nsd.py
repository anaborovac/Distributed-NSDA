
import torch 
import scipy.special
import numpy as np


def predict(device, model, data, mini_batch = 128):
	"""
	Prediction:
		device - cuda or cpu
		model - trained NSD
		data - array of size (N, *) to be predicted
		mini_batch - the amount of data predicted at the same time

	Output:
		array of size N with predicted probabbilities
	"""
	
	model.eval()
	model.to(device)

	(n_batch, _, _) = data.shape

	p = np.zeros(n_batch)

	for t in range(0, n_batch, mini_batch):
		t_end = min(t + mini_batch, n_batch)

		data_segment = torch.tensor(data[t:t_end, :, :]).float().to(device)
		score = model(data_segment).detach().cpu().numpy()
		probabilities = scipy.special.softmax(score, axis = 1)

		p[t:t_end] = probabilities[:, 1]

	return p


