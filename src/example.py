
import torch
import numpy as np

import data_preprocessing as DP 
import train_nsd as T 
import predict_nsd as P 
import dawid_skene as DS  
import logistic_regression as LR
import aux_functions as AF
import seizure_events as SE
import metrics as M


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


# 2.1) Train the NSDs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

data_train_1 = T.prepare_data(data_1['Data'], data_1['Y'])
model_1 = T.train(device, data_train_1)

data_train_2 = T.prepare_data(data_2['Data'], data_2['Y'])
model_2 = T.train(device, data_train_2)

data_train_3_x, data_train_3_y = T.prepare_data(data_3['Data'], data_3['Y'])
model_3 = T.train(device, (data_train_3_x, data_train_3_y))


# 2.2) Train the Logistic regression classifier

data_train_3_lr = np.zeros((len(data_train_3_x), 2))
data_train_3_lr[:, 0] = P.predict(device, model_1, data_train_3_x)
data_train_3_lr[:, 1] = P.predict(device, model_2, data_train_3_x)
lr = LR.train_lr((data_train_3_lr, data_train_3_y))


# 3) Make predictions on the test data 

data_test_x = data_test['Data']
predictions_local = np.zeros((len(data_test_x), 3))
predictions_local[:, 0] = P.predict(device, model_1, data_test_x)
predictions_local[:, 1] = P.predict(device, model_2, data_test_x)
predictions_local[:, 2] = P.predict(device, model_3, data_test_x)


# 4) Aggregate predictions

predictions_mv = np.mean(AF.round_probabilities(predictions_local), axis = 1)
predictions_mean = np.mean(predictions_local, axis = 1)
predictions_wm = LR.predict_lr(lr, predictions_local[:, :2])
predictions_ds, _, _, _, _ = DS.run_dawid_skene(predictions_local)


# 5) Find seizure events 

seizures_mv = SE.get_seizure_intervals(data_test['TimeStamps'], segment_duration, segment_overlap, predictions_mv, ignore_short = True)
seizures_mean = SE.get_seizure_intervals(data_test['TimeStamps'], segment_duration, segment_overlap, predictions_mean, ignore_short = True)
seizures_wm = SE.get_seizure_intervals(data_test['TimeStamps'], segment_duration, segment_overlap, predictions_wm, ignore_short = True)
seizures_ds = SE.get_seizure_intervals(data_test['TimeStamps'], segment_duration, segment_overlap, predictions_ds, ignore_short = True)


# 6) Calculate metrics

y = data_test['Y']

auc_mv, se_mv, sp_mv = M.calculate_segment_metrics(y[y != 2], AF.round_probabilities(predictions_mv[y != 2]), predictions_mv[y != 2]) # ignore segments with disagreements
sdr_mv, fd_mv, mfdd_mv = M.calculate_event_metrics(data_test['Duration'], data_test['Seizures_union'], data_test['Seizures_intersection'], seizures_mv)
print(f'Majority vote: AUC {auc_mv:.2f}, SE {se_mv:.0f}%, SP {sp_mv:.0f}%, SDR {sdr_mv:.0f}%, FD {fd_mv:.2f}/h, MFDD {mfdd_mv:.2f}s')

auc_mean, se_mean, sp_mean = M.calculate_segment_metrics(y[y != 2], AF.round_probabilities(predictions_mean[y != 2]), predictions_mean[y != 2])
sdr_mean, fd_mean, mfdd_mean = M.calculate_event_metrics(data_test['Duration'], data_test['Seizures_union'], data_test['Seizures_intersection'], seizures_mean)
print(f'Mean: AUC {auc_mean:.2f}, SE {se_mean:.0f}%, SP {sp_mean:.0f}%, SDR {sdr_mean:.0f}%, FD {fd_mean:.2f}/h, MFDD {mfdd_mean:.2f}s')

auc_wm, se_wm, sp_wm = M.calculate_segment_metrics(y[y != 2], AF.round_probabilities(predictions_wm[y != 2]), predictions_wm[y != 2])
sdr_wm, fd_wm, mfdd_wm = M.calculate_event_metrics(data_test['Duration'], data_test['Seizures_union'], data_test['Seizures_intersection'], seizures_wm)
print(f'Weighted mean: AUC {auc_wm:.2f}, SE {se_wm:.0f}%, SP {sp_wm:.0f}%, SDR {sdr_wm:.0f}%, FD {fd_wm:.2f}/h, MFDD {mfdd_wm:.2f}s')

auc_ds, se_ds, sp_ds = M.calculate_segment_metrics(y[y != 2], AF.round_probabilities(predictions_ds[y != 2]), predictions_ds[y != 2])
sdr_ds, fd_ds, mfdd_ds = M.calculate_event_metrics(data_test['Duration'], data_test['Seizures_union'], data_test['Seizures_intersection'], seizures_ds)
print(f'Dawid-Skeneat: AUC {auc_ds:.2f}, SE {se_ds:.0f}%, SP {sp_ds:.0f}%, SDR {sdr_ds:.0f}%, FD {fd_ds:.2f}/h, MFDD {mfdd_ds:.2f}s')

