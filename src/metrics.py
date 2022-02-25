
import numpy as np  
import sklearn.metrics as m 


def calculate_sdr(target_seizures, pred_seizures):

	if len(pred_seizures) == 0:
		return 0

	sdr = 0 # number of detected seizures

	for (s, e) in target_seizures:
		tmp_pred = pred_seizures[np.logical_and( pred_seizures[:, 0] < e, pred_seizures[:, 1] > s)]

		if tmp_pred.shape[0] != 0:
			sdr += 1 

	return  sdr / target_seizures.shape[0]


def calculate_fd(duration, target_seizures, pred_seizures):

	fd = 0 # number of false detected seizures
	fdd = [] # durations of false detected seizures

	if len(target_seizures) == 0:
		return len(pred_seizures), [(e-s) for (s, e) in pred_seizures]

	for (s, e) in pred_seizures:
		tmp_target = target_seizures[np.logical_and( target_seizures[:, 0] <= e, target_seizures[:, 1] >= s)]

		if tmp_target.shape[0] == 0:
			fd += 1 
			fdd += [(e-s)] 

	mfdd = np.mean(fdd) if len(fdd) != 0 else 0

	return  fd / duration, mfdd


def calculate_event_metrics(duration, intersection_s, union_s, pred_s):

	sdr = calculate_sdr(intersection_s, pred_s)
	fd, mfdd = calculate_fd(duration, union_s, pred_s)

	return sdr * 100, fd, mfdd


def calculate_segment_metrics(target, prediction, probabilities):

	((tn, fp), (fn, tp)) = m.confusion_matrix(target, prediction)

	auc = None if probabilities is None else m.roc_auc_score(target, probabilities)
	se = (tp / (tp + fn)) * 100
	sp = (tn / (tn + fp)) * 100
	
	return auc, se, sp



