
import numpy as np 
import mne
import scipy.signal
import pickle
from os import path

import seizure_events as SE
import aux_functions as AF


def channel_preprocessing(fs, data, low_cut = 0.5, high_cut = 32, fs_final = 32):
	""" 
	Parameters:
		fs - sampling frequency of input data
		data - data to be preprocessed, array of shape N
		low_cut - low-cut frequency in Hz 
		high_cut - hight-cut frequency in Hz
		fs_final - sampling frequency of preprocessed data

	Output:
		preprocessed data
	""" 

	b, a = scipy.signal.cheby2(6, 80, [low_cut / (fs / 2), high_cut / (fs / 2)], btype = 'bandpass')
	preprocessed = scipy.signal.lfilter(b, a, data)
	preprocessed = scipy.signal.resample_poly(preprocessed, fs_final, fs)

	preprocessed[preprocessed > 200] = 200 
	preprocessed[preprocessed < -200] = -200 

	preprocessed = preprocessed / 400 * (2**16 - 1) + (2**15) + 0.5
	
	# uint16
	preprocessed[preprocessed > 65535] = 65535
	preprocessed[preprocessed < 0] = 0

	preprocessed = np.round(preprocessed) - 32768

	return preprocessed


def get_seizures_one_expert(pid, annotations):
	"""
	Parameters:
		pid - patient ID 
		annotations - one of csv files with annotations (https://zenodo.org/record/4940267#.YhkHDJPP01I)

	Output:
		array of shape (*, 2) with seizure time intervals
	"""

	seizures = np.genfromtxt(annotations, delimiter = ',')
	seizures = seizures[:, seizures[0] == pid][1:, 0]
	seizures = np.where(seizures == 1)[0]
	seizures = np.split(seizures, np.where(np.diff(seizures) != 1)[0]+1)
	seizures = [[s[0], s[-1]] for s in seizures]

	return np.array(seizures)


def get_seizures(pid):
	"""
	Parameters:
		pid - patient ID 

	Output:
		union of seizures - array of shape (*, 2) with time intervals of union of all annotated seizures 
		intersection of seizures - array of shape (*, 2) with time intervals of consensus seizures
	"""

	seizures_A = get_seizures_one_expert(pid, '../data/annotations_2017_A.csv')
	seizures_B = get_seizures_one_expert(pid, '../data/annotations_2017_B.csv')
	seizures_C = get_seizures_one_expert(pid, '../data/annotations_2017_C.csv')

	union_seizures = SE.intervals_union(seizures_A, seizures_B)
	union_seizures = SE.intervals_union(union_seizures, seizures_C)

	intersection_seizures = SE.intervals_intersection(seizures_A, seizures_B)
	intersection_seizures = SE.intervals_intersection(intersection_seizures, seizures_C)

	return union_seizures, intersection_seizures


def get_labels(ts, duration, seizures_union, seizure_intersection):
	"""
	Parameters:
		ts - array os size N1 with segment time stamps
		duration - duration of a recording
		seizure_union - array of shape (N2, 2) with time intervals of union of all annotated seizures
		seizure_intersection - array of shape (N3, 2) with time intervals of consensus seizures
		
	Output:
		array of size N1 with target values (0 - non-seizure, 1 - seizure, 2 - segment with disagreements or partly seizure/non-seizure segment)
	"""

	y = np.zeros(len(ts))

	for i, s in enumerate(ts):
		e = s + duration 

		if seizure_intersection[np.logical_and(seizure_intersection[:, 0] <= s, seizure_intersection[:, 1] >= e)].shape[0] != 0:
			# if the current interval is inside of one of inteserction seizures
			y[i] = 1 
		elif seizures_union[np.logical_and(seizures_union[:, 0] < e, seizures_union[:, 1] > s)].shape[0] != 0:
			# if the currect interval intersects with a seizure from any expert
			y[i] = 2

	return y


def data_preprocessing(pid, file_name, montage, segment_duration = 16, segment_overlap = 12, fs_final = 32, save = True):
	"""
	Parameters:
		pid - patient ID 
		file_name - edf file name located in data/ folder
		montage - tuple (active, reference) where active and reference and arrays of size N with active and reference electrodes 
		segment_duration - duration of segments in seconds the recording is cut into
		segment_overlap - size of overlap in seconds of two consequtive segments
		fs_final - sampling frequency of output data
		save - True if preprocessed data is saved into data_preprocessed/ folder

	Output: 
		dictionary with 
			Data - preprocessed data, array of size (*, N, *)
			TimeStamps - array with time stamps of segments in Data
			Y - array with target values for segments in Data
			Seizures_union - array of shape (*, 2) with time intervals of union of all annotated seizures
			Seizures_intersection - array of shape (*, 2) with time intervals of consensus seizures
			Duration - recording duration
	"""

	file_name_preprocessed = file_name.replace('.edf', '_preprocessed.pt')
	file_name_preprocessed = f'../data_preprocessed/{file_name_preprocessed}'

	if path.exists(file_name_preprocessed):
		return AF.load_data(file_name_preprocessed)

	(active, reference) = montage

	data = mne.io.read_raw_edf(f'../data/{file_name}', infer_types = True)
	raw_data = data.get_data(units = {'eeg': 'uV'})
	channels = [c.replace('Ref', 'REF') for c in data.ch_names]
	fs = int(data.info['sfreq'])
	duration = raw_data.shape[1] / fs / 60 / 60 # in hours

	data_montage = np.zeros((len(active), raw_data.shape[1]))
	for i, (a, r) in enumerate(zip(active, reference)):
		data_montage[i] = raw_data[channels.index(f'{a}-REF')] - raw_data[channels.index(f'{r}-REF')]

	data_segments, time_stamps = [], []
	step_size = (segment_duration - segment_overlap) * fs
	final_point = data_montage.shape[1] - segment_duration * fs

	for start_point in range(0, final_point, step_size):
		end_point = start_point + segment_duration * fs

		current_segment = data_montage[:, start_point:end_point]
		preprocessed_segment = np.zeros((current_segment.shape[0], segment_duration * fs_final))
		ignore = False
		for channel in current_segment:
			preprocessed_segment[i] = channel_preprocessing(fs, channel, fs_final = fs_final)
			ignore = True if (channel.size - np.count_nonzero(channel)) > fs else ignore

		if ignore: continue

		data_segments.append(preprocessed_segment)
		time_stamps.append(start_point / fs)

	union_seizures, intersection_seizures = get_seizures(pid)
	y = get_labels(time_stamps, segment_duration, union_seizures, intersection_seizures)

	D = {'Data': np.array(data_segments), 'TimeStamps': np.array(time_stamps), 'Y': y, 'Seizures_union': union_seizures, 'Seizures_intersection': intersection_seizures, 'Duration': duration}

	if save:
		AF.save_data(file_name_preprocessed, D)

	return D





