
import torch
import numpy as np

import data_preprocessing as DP 
import train_nsd as T 
import predict_nsd as P 
import dawid_skene as DS  


# 1) Data loading and preprocessing
# In This example each patient represent one data set

active = ['Fp2', 'F4', 'C4', 'P4', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F8', 'T4', 'T6', 'Fp1', 'F7', 'T3', 'T5', 'Fz', 'Cz'] 
reference = ['F4', 'C4', 'P4', 'O2', 'F3', 'C3', 'P3', 'O1', 'F8', 'T4', 'T6', 'O2', 'F7', 'T3', 'T5', 'O1', 'Cz', 'Pz'] 

segment_duration = 16
segment_overlap = 12

data_1 = DP.data_preprocessing(1, 'eeg1.edf', (active, reference), segment_duration = segment_duration, segment_overlap = segment_overlap)
data_2 = DP.data_preprocessing(4, 'eeg4.edf', (active, reference), segment_duration = segment_duration, segment_overlap = segment_overlap)
data_3 = DP.data_preprocessing(5, 'eeg5.edf', (active, reference), segment_duration = segment_duration, segment_overlap = segment_overlap)

data_test = DP.data_preprocessing(7, 'eeg7.edf', (active, reference), segment_duration = segment_duration, segment_overlap = segment_overlap)


# 2) Train the NSDs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

data_train_1 = T.prepare_data(data_1['Data'], data_1['Y'])
model_1 = T.train(device, data_train_1)

data_train_2 = T.prepare_data(data_2['Data'], data_2['Y'])
model_2 = T.train(device, data_train_2)

data_train_3 = T.prepare_data(data_3['Data'], data_3['Y'])
model_3 = T.train(device, data_train_3)


# 3) Make predictions on the test data 

data_test_x = data_test['Data']
predictions_local = np.zeros((len(data_test_x), 3))
predictions_local[:, 0] = P.predict(device, model_1, data_test_x)
predictions_local[:, 1] = P.predict(device, model_2, data_test_x)
predictions_local[:, 2] = P.predict(device, model_3, data_test_x)


# 4) Aggregate predictions

predictions_ds, _, _, _, _ = DS.run_dawid_skene(predictions_local)
predictions_mean = np.mean(predictions_local, axis = 1)


# 5) Find seizure events 

# 6) Calculate metrics 

# 7) Plot results 