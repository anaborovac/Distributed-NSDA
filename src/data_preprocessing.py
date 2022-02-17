
import numpy as np 
import mne
import scipy.signal
import pickle


def channel_preprocessing(fs, data, low_cut = 0.5, high_cut = 32, fs_final = 32):

	b, a = scipy.signal.cheby2(6, 80, [low_cut / (fs / 2), high_cut / (fs / 2)], btype = 'bandpass')
	dum = scipy.signal.lfilter(b, a, data)
	dum = scipy.signal.resample_poly(dum, fs_final, fs)

	dum[dum > 200] = 200 
	dum[dum < -200] = -200 

	dum = dum / 400 * (2**16 - 1) + (2**15) + 0.5
	
	# uint16
	dum[dum > 65535] = 65535
	dum[dum < 0] = 0

	fn = np.round(dum) - 32768

	return fn


def data_preprocessing(file_name, montage, segment_duration = 16, segment_overlap = 12, fs_final = 32, save = True):

	(active, reference) = montage

	data = mne.io.read_raw_edf(f'../data/{file_name}', infer_types = True, exclude = ['EKG-REF', 'Effort-REF'])
	raw_data = data.get_data(units = {'eeg': 'uV'})
	channels = data.ch_names
	fs = int(data.info['sfreq'])

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

	return data_segments


def example_run():

	active = ['Fp2', 'F4', 'C4', 'P4', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F8', 'T4', 'T6', 'Fp1', 'F7', 'T3', 'T5', 'Fz', 'Cz'] 
	reference = ['F4', 'C4', 'P4', 'O2', 'F3', 'C3', 'P3', 'O1', 'F8', 'T4', 'T6', 'O2', 'F7', 'T3', 'T5', 'O1', 'Cz', 'Pz'] 

	data = data_preprocessing('eeg1.edf', (active, reference))

