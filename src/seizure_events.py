
import numpy as np 


def merge_overlaps(s):

	if s.shape[0] < 2:
		return s

	tmp_s = s.copy()

	# sort events by starting time stamp
	sorted_s = tmp_s[np.argsort(tmp_s[:, 0])]

	i = 1 
	while i < sorted_s.shape[0]:
		if sorted_s[i, 0] >= sorted_s[i-1, 0] and sorted_s[i, 0] <= sorted_s[i-1, 1]:
			sorted_s[i-1, 1] = max(sorted_s[i, 1], sorted_s[i-1, 1])
			sorted_s = np.delete(sorted_s, i, axis = 0)
		else:
			i += 1

	return sorted_s


def remove_empty_seizures(s):

	if len(s) == 0:
		return s

	new_s = s.copy()
	empy_seizures = np.where((new_s[:, 1] - new_s[:, 0]) == 0)[0]
	new_s = np.delete(new_s, empy_seizures, axis = 0)

	return new_s


def union_seizures(s, additional_seizures):

	if len(s) == 0:
		return additional_seizures

	new_s = np.copy(s)
	for (start, end) in additional_seizures:

		c = np.logical_and( new_s[:, 0] < end
						  , new_s[:, 1] > start)

		i = np.where(c)[0][0] if new_s[c].shape[0] != 0 else None
			
		if i is not None:
			new_s[i, 0] = min(new_s[i, 0], start)
			new_s[i, 1] = max(new_s[i, 1], end)
		else:
			new_s = np.r_[new_s, [[start, end]]].reshape(-1, 2)

	return remove_empty_seizures(merge_overlaps(new_s))


def intersection_seizures(s, additional_seizures):

	new_s = []
	for (start, end) in s:

		c = np.logical_and( additional_seizures[:, 0] <= end
						  , additional_seizures[:, 1] >= start)

		new_s += [[max(s, start), min(e, end)] for (s, e) in additional_seizures[c]]

	return remove_empty_seizures(np.array(new_s))


def calculate_sdr_fd(intersection_s, union_s, p, seconds = False):

	detected_seizures, coverage_seizures = calculate_sdr(intersection_s, p)
	detected_false_seizures, coverage_false_seizures = calculate_fd(union_s, p, seconds)

	return {'TP': detected_seizures, 'TP_C': coverage_seizures, 'FP': detected_false_seizures, 'FP_C': coverage_false_seizures}


def calculate_sdr(target_seizures, pred_seizures):

	d_seizures = 0 # number of detected seizures
	c_seizures_all = [] # coverage per seizure

	if len(pred_seizures) == 0:
		return 0, [0 for _ in range(len(target_seizures))]

	for (s, e) in target_seizures:
		tmp_pred = pred_seizures[np.logical_and( pred_seizures[:, 0] < e
											   , pred_seizures[:, 1] > s)]
		tmp_pred[tmp_pred[:, 0] < s, 0] = s
		tmp_pred[tmp_pred[:, 1] > e, 1] = e

		if tmp_pred.shape[0] != 0:
			d_seizures += 1 
			c_seizures_all += [np.sum(tmp_pred[:, 1] - tmp_pred[:, 0]) / (e-s)]
		else:
			c_seizures_all += [0]

	return  d_seizures, c_seizures_all


def calculate_fd(target_seizures, pred_seizures, seconds = False):

	d_f_seizures = 0 # number of false detected seizures
	c_f_seizures = [] # time of false detected seizures

	second_1 = C.interval_end(0, 1, seconds)

	if len(target_seizures) == 0:
		return len(pred_seizures), [(e-s) / second_1 for (s, e) in pred_seizures]

	for (s, e) in pred_seizures:
		tmp_target = target_seizures[np.logical_and( target_seizures[:, 0] <= e
											       , target_seizures[:, 1] >= s)]

		if tmp_target.shape[0] == 0:
			print(s, e)
			d_f_seizures += 1 
			c_f_seizures += [(e-s) / second_1] # /10**3 to get seconds

	return  d_f_seizures, c_f_seizures


def get_seizure_intervals(ts, pred, ignore_short = False, seconds = False):

	P = dict()
	for t in ts:
		tmp_pred = pred[ts == t][0]
		for tt in range(4):
			k = C.interval_end(t, tt*4, seconds)
			P.setdefault(k, [])
			P[k] += [tmp_pred]

	ts_small = list(P.keys())
	ts_small.sort()

	time_segments = DA.get_time_intervals(ts_small.copy(), 12, 16, seconds)

	pred_seizures = []
	for tmp_ts in time_segments:
		pred_final = np.round([np.mean(P[t]) for t in tmp_ts])

		seizures_start = [tmp_ts[0]] if pred_final[0] == 1 else []
		seizures_start += [tmp_ts[i] for i in range(1, len(pred_final)) if pred_final[i] == 1 and pred_final[i-1] == 0]

		seizures_end = [tmp_ts[i] for i in range(0, len(pred_final)-1) if pred_final[i] == 1 and pred_final[i+1] == 0]
		seizures_end += [tmp_ts[len(pred_final) - 1]] if pred_final[-1] == 1 else []

		pred_seizures += [[s, C.interval_end(e, 4, seconds)] for (s, e) in zip(seizures_start, seizures_end)]

	if ignore_short:
		min_duration = C.interval_end(0, 10, seconds)
		pred_seizures = [[s, e] for (s, e) in pred_seizures if (e - s) >= min_duration]

	return  np.array(pred_seizures)