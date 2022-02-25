
import numpy as np  
import sklearn.metrics as m 


def calculate_sdr(seizure_intersection, predicted_seizures):
	"""
	Parameters:
		seizure_intersection - array of shape (N1, 2) with time intervals of consensus seizures
		predicted_seizures - array of shape (N2, 2) with time intervals of predicted seizures

	Output:
		seirure detection rate in percentages
	"""

	if len(predicted_seizures) == 0:
		return 0

	sdr = 0 # number of detected seizures

	for (s, e) in seizure_intersection:
		tmp_pred = predicted_seizures[np.logical_and(predicted_seizures[:, 0] < e, predicted_seizures[:, 1] > s)]

		if tmp_pred.shape[0] != 0:
			sdr += 1 

	return  sdr / seizure_intersection.shape[0] * 100


def calculate_fd(duration, seizures_union, predicted_seizures):
	"""
	Parameters:
		duration - duration of test data (e.g. recording) in hours
		seizure_union - array of shape (N1, 2) with time intervals of union of all annotated seizures
		predicted_seizures - array of shape (N2, 2) with time intervals of predicted seizures

	Output:
		false detections per hour, mean false detection duration
	"""

	fd = 0 # number of false detected seizures
	fdd = [] # durations of false detected seizures

	if len(seizures_union) == 0:
		return len(predicted_seizures), [(e-s) for (s, e) in predicted_seizures]

	for (s, e) in predicted_seizures:
		tmp_target = seizures_union[np.logical_and(seizures_union[:, 0] <= e, seizures_union[:, 1] >= s)]

		if tmp_target.shape[0] == 0:
			fd += 1 
			fdd += [(e-s)] 

	mfdd = np.mean(fdd) if len(fdd) != 0 else 0

	return  fd / duration, mfdd


def calculate_event_metrics(duration, seizures_union, seizures_intersection, predicted_seizures):
	"""
	Parameters:
		duration - duration of test data (e.g. recording) in hours
		seizure_union - array of shape (N1, 2) with time intervals of union of all annotated seizures
		seizure_intersection - array of shape (N2, 2) with time intervals of consensus seizures
		predicted_seizures - array of shape (N3, 2) with time intervals of predicted seizures

	Output:
		seirure detection rate, false detections per hour, mean false detection duration
	"""

	sdr = calculate_sdr(seizures_intersection, predicted_seizures)
	fd, mfdd = calculate_fd(duration, seizures_union, predicted_seizures)

	return sdr, fd, mfdd


def calculate_segment_metrics(target, predictions, probabilities):
	"""
	Parameters:
		target - array of size N with target values (0/1)
		predictions - array of size N with predictions (0/1)
		probabilities - array of size N with probabilities associated with predicitons (0-1)

	Output:
		AUC
		sensitivity in percentages
		specificity in percentages
	"""

	((tn, fp), (fn, tp)) = m.confusion_matrix(target, prediction)

	auc = None if probabilities is None else m.roc_auc_score(target, probabilities)
	se = (tp / (tp + fn)) * 100
	sp = (tn / (tn + fp)) * 100
	
	return auc, se, sp



