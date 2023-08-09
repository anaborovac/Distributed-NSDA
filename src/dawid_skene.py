
import numpy as np

import aux_functions as AF


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


def run_dawid_skene(probabilities, epsilon = 1e-5, kmax = 5000):
	"""
	Dawid, Alexander Philip, and Allan M. Skene. "Maximum likelihood estimation of observer error‐rates using the EM algorithm." Journal of the Royal Statistical Society: Series C (Applied Statistics) 28.1 (1979): 20-28.

	Parameters:
		probabilities - array of size (N1, N2) with probabilities for N1 segments made by N2 scorers
		epsilon - stopping critera
		kmax - maximum number of interations

	Output:
		T - consensus labels estimated by the Dawid-Skene method
		pi - array of size (N2, 2) with estimated specificity and sensitivity values for each scorer 
		likelihood - value of log-likelihood function after convergence is reached or number of interation is equal to kmax
		i - number of interation steps
		diff - difference in log-likelihood of the last two steps
	"""

	labels = AF.round_probabilities(probabilities)
	
	T = init_T(probabilities) 
	p = estimate_p(T)
	pi = estimate_pi(T, labels)
	a, b = calculate_a_b(pi, labels)
	loss_old = calculate_log_likelihood(p, a, b)
	
	i = 1

	while True:
		T = estimate_T(p, a, b)
		p = estimate_p(T)
		pi = estimate_pi(T, labels)
		a, b = calculate_a_b(pi, labels)
		likelihood_old = calculate_log_likelihood(p, a, b)
		
		diff = np.abs(loss_old - likelihood_old)

		if (diff < epsilon) or (i == kmax):
			return T, pi, likelihood_old, i, diff
		
		loss_old = likelihood_old.copy()
		i += 1
	


	


