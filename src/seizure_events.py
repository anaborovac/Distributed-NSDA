
import numpy as np 

import aux_functions as AF


def merge_overlaps(intervals):
	"""
	Parameters:
		intervals - array of size (N, 2) with time intervals

	Output:
		array of size (*, 2) in which no intervals overlap
	"""

	if intervals.shape[0] < 2:
		return intervals

	# sort intervals by starting time stamp
	sorted_intervals = intervals[np.argsort(intervals[:, 0])]

	i = 1 
	while i < sorted_intervals.shape[0]:
		if sorted_intervals[i, 0] >= sorted_intervals[i-1, 0] and sorted_intervals[i, 0] <= sorted_intervals[i-1, 1]:
			# in case two consequtive intervals overlap
			sorted_intervals[i-1, 1] = max(sorted_intervals[i, 1], sorted_intervals[i-1, 1])
			sorted_intervals = np.delete(sorted_intervals, i, axis = 0)
		else:
			i += 1

	return sorted_intervals


def remove_empty_intervals(intervals):
	"""
	Parameters:
		intervals - array of size (N, 2) with time intervals

	Output:
		array of size (*, 2) with no time intervals with duration 0
	"""

	if len(s) == 0:
		return s

	new_intervals = intervals.copy()
	empy_intervals = np.where((intervals[:, 1] - intervals[:, 0]) == 0)[0]
	new_intervals = np.delete(new_intervals, empy_seizures, axis = 0)

	return new_intervals


def intervals_union(intervals_1, intervals_2):
	"""
	Parameters:
		seizures - array of size (N1, 2) with seizure time intervals
		additional_seizures - array of size (N2, 2) with seizure time intervals

	Output:
		array of shape (*, 2) with union intervals of input arrays
	"""

	if len(intervals_1) == 0:
		return intervals_2

	union = np.copy(intervals_1)
	for (start, end) in intervals_2:

		c = np.logical_and(union[:, 0] < end, union[:, 1] > start)
			
		if union[c].shape[0] != 0:
			# in case of overlap, make a union of two intervals
			i = np.where(c)[0][0]
			union[i, 0] = min(union[i, 0], start)
			union[i, 1] = max(union[i, 1], end)
		else:
			# in case there does not exist an overlap, just add the interval to the union
			union = np.r_[union, [[start, end]]].reshape(-1, 2)

	return remove_empty_intervals(merge_overlaps(union))


def intervals_intersection(intervals_1, intervals_2):
	"""
	Parameters:
		seizures - array of size (N1, 2) with seizure time intervals
		additional_seizures - array of size (N2, 2) with seizure time intervals

	Output:
		array of shape (*, 2) with intersection intervals of input arrays
	"""

	intersection = []
	for (start, end) in intervals_1:

		c = np.logical_and(intervals_2[:, 0] <= end, intervals_2[:, 1] >= start)

		intersection += [[max(s, start), min(e, end)] for (s, e) in intervals_2[c]]

	return remove_empty_intervals(np.array(intersection))


def get_seizure_intervals(ts, segment_duration, segment_overlap, predictions, ignore_short = False):
	"""
	Parameters:
		ts - array of size N with time stamps
		segment_duration - duration of each segment in seconds
		segment_overlap - overlap in seconds of two consequitve segments
		predictions - array of size N with predictions
		ignore_short - True if predicted seizures shorter than 10 seconds are not included in the output

	Output:
		array of size (*, 2) of time intervals where seizures are predicted
	"""


	time_stap = segment_duration - segment_overlap

	P = dict()
	for t in ts.astype(int):
		pred_t = predictions[ts == t][0]
		for tt in range(t, t + segment_duration + 1, time_stap):
			P.setdefault(tt, [])
			P[tt] += [pred_t]

	ts_small = list(P.keys())
	ts_small.sort()

	time_segments = np.split(ts_small, np.where(np.diff(ts_small) != time_stap)[0]+1)

	pred_seizures = []
	for time_seg in time_segments:
		pred_final = AF.round_probabilities([np.mean(P[t]) for t in time_seg])

		seizures_start = [time_seg[0]] if pred_final[0] == 1 else []
		seizures_start += [time_seg[i] for i in range(1, len(pred_final)) if pred_final[i] == 1 and pred_final[i-1] == 0]

		seizures_end = [time_seg[i] for i in range(0, len(pred_final)-1) if pred_final[i] == 1 and pred_final[i+1] == 0]
		seizures_end += [time_seg[len(pred_final) - 1]] if pred_final[-1] == 1 else []

		pred_seizures += [[s, e + time_stap] for (s, e) in zip(seizures_start, seizures_end)]

	if ignore_short:
		min_duration = 10 # seconds
		pred_seizures = [[s, e] for (s, e) in pred_seizures if (e - s) >= min_duration]

	return  np.array(pred_seizures)