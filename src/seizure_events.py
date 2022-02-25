
import numpy as np 

import aux_functions as AF


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


def get_seizure_intervals(time_stamps, interval_duration, overlap_duration, predictions, ignore_short = False):

	time_stap = interval_duration - overlap_duration

	P = dict()
	for t in time_stamps.astype(int):
		pred_t = predictions[time_stamps == t][0]
		for tt in range(t, t + interval_duration + 1, time_stap):
			P.setdefault(tt, [])
			P[tt] += [pred_t]

	ts_small = list(P.keys())
	ts_small.sort()

	time_segments = np.split(ts_small, np.where(np.diff(ts_small) != time_stap)[0]+1)

	pred_seizures = []
	for tmp_ts in time_segments:
		pred_final = AF.round_probabilities([np.mean(P[t]) for t in tmp_ts])

		seizures_start = [tmp_ts[0]] if pred_final[0] == 1 else []
		seizures_start += [tmp_ts[i] for i in range(1, len(pred_final)) if pred_final[i] == 1 and pred_final[i-1] == 0]

		seizures_end = [tmp_ts[i] for i in range(0, len(pred_final)-1) if pred_final[i] == 1 and pred_final[i+1] == 0]
		seizures_end += [tmp_ts[len(pred_final) - 1]] if pred_final[-1] == 1 else []

		pred_seizures += [[s, e+4] for (s, e) in zip(seizures_start, seizures_end)]

	if ignore_short:
		min_duration = 10 # seconds
		pred_seizures = [[s, e] for (s, e) in pred_seizures if (e - s) >= min_duration]

	return  np.array(pred_seizures)