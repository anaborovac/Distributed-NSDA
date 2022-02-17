
import numpy as np

import aux_funcitons as AF


def init_T(labels):
	return np.mean(labels, axis = 1)


def estimate_p(T):
	return np.mean(T)


def estimate_pi(T, labels):

	positives = max(np.sum(T), 1e-5)
	negatives = max(np.sum(1 - T), 1e-5)

	pi = np.ones((labels.shape[1], 2))
	for i in range(labels.shape[1]): # iterate over the experts
		labels_expert = labels[:, i]

		pi[i, 0] = np.sum(np.multiply(1 - T, 1 - labels_expert)) / negatives # sp
		pi[i, 1] = np.sum(np.multiply(T, labels_expert)) / positives # se

	return pi


def calculate_a_b(pi, labels):
	a, b = np.zeros(labels.shape[0]), np.zeros(labels.shape[0])
	for i in range(labels.shape[0]):
		labels_instance = labels[i]
		
		b[i] = np.prod((pi[:, 0] ** (1 - labels_instance)) * ((1 - pi[:, 0]) ** labels_instance))
		a[i] = np.prod((pi[:, 1] ** labels_instance) * ((1 - pi[:, 1]) ** (1 - labels_instance)))

	return a, b


def estimate_T(p, a, b):
	return (a * p) / (a * p + b * (1 - p))


def calculate_log_likelihood(p, a, b):
	return np.sum(np.log(a * p + b * (1 - p)))


def run_dawid_skene(labels_p, epsilon = 1e-5, kmax = 5000):

	labels = AF.round_probabilities(labels_p)
	
	T = init_T(labels_p) 
	p = estimate_p(T)
	pi = estimate_pi(T, labels)
	a, b = calculate_a_b(pi, labels)
	l_old = calculate_log_likelihood(p, a, b)
	
	i = 1

	while True:
		T = estimate_T(p, a, b)
		p = estimate_p(T)
		pi = estimate_pi(T, labels)
		a, b = calculate_a_b(pi, labels)
		l = calculate_log_likelihood(p, a, b)
		
		diff = np.abs(l_old - l)

		if (diff < epsilon) or (i == kmax):
			return T, pi, l, i, diff
		
		l_old = l.copy()
		i += 1
	

def example_run(): 

	n, n_expert = 100, 5

	T, _, _, _, _ = run_dawid_skene(np.random.random(size = (n, n_expert)))

	print(T.shape)
	


