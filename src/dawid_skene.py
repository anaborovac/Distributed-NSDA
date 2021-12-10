
import numpy as np


def round_probabilities(x):

	tmp_x = np.copy(x)
	tmp_x[tmp_x >= 0.5] = 1 
	tmp_x[tmp_x < 0.5] = 0

	return tmp_x


def init_T(labels):
	return np.mean(labels, axis = 1)


def estimate_p(T):
	return np.mean(T)


def estimate_pi(T, labels):

	positives = max(np.sum(T), 1e-5)
	negatives = max(np.sum(1 - T), 1e-5)

	pi = np.ones((labels.shape[1], 2))
	for i in range(labels.shape[1]): # iterate over the experts
		tmp_labels = labels[:, i]

		pi[i, 0] = np.sum(np.multiply(1 - T, 1 - tmp_labels)) / negatives # sp
		pi[i, 1] = np.sum(np.multiply(T, tmp_labels)) / positives # se

	return pi


def calculate_a_b(pi, labels):
	a, b = np.zeros(labels.shape[0]), np.zeros(labels.shape[0])
	for i in range(labels.shape[0]):
		tmp_labels = labels[i]
		tmp_pi = pi
		
		b[i] = np.prod((tmp_pi[:, 0] ** (1 - tmp_labels)) * ((1 - tmp_pi[:, 0]) ** tmp_labels))
		a[i] = np.prod((tmp_pi[:, 1] ** tmp_labels) * ((1 - tmp_pi[:, 1]) ** (1 - tmp_labels)))

	return a, b


def estimate_T(p, a, b):
	return (a * p) / (a * p + b * (1 - p))


def calculate_likelihood(p, a, b):
	return np.sum(np.log(a * p + b * (1 - p)))


def run_dawid_skene(labels_p):

	labels = round_probabilities(labels_p)
	
	T = init_T(labels_p) 
	p = estimate_p(T)
	pi = estimate_pi(T, labels)
	a, b = calculate_a_b(pi, labels)
	l_old = calculate_likelihood(p, a, b)
	
	i = 1

	while True:
		T = estimate_T(p, a, b)
		p = estimate_p(T)
		pi = estimate_pi(T, labels)
		a, b = calculate_a_b(pi, labels)
		l = calculate_likelihood(p, a, b)
		
		diff = np.abs(l_old - l)

		if (diff < 1e-5) or (i == 5000):
			return T, pi, l, i, diff
		
		l_old = l.copy()
		i += 1
	

def example_run(): 

	n, n_expert = 100, 5

	T, _, _, _, _ = run_dawid_skene(np.random.random(size = (100, 5)))

	print(T.shape)
	


