

import data_preprocessing as DP 
import train_nsd as T 
import predict_nsd as P 
import dawid_skene as DS  


# 1) Data loading and preprocessing

active = ['Fp2', 'F4', 'C4', 'P4', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F8', 'T4', 'T6', 'Fp1', 'F7', 'T3', 'T5', 'Fz', 'Cz'] 
reference = ['F4', 'C4', 'P4', 'O2', 'F3', 'C3', 'P3', 'O1', 'F8', 'T4', 'T6', 'O2', 'F7', 'T3', 'T5', 'O1', 'Cz', 'Pz'] 

data_train_1 = data_preprocessing('eeg1.edf', (active, reference))
data_train_2 = data_preprocessing('eeg1.edf', (active, reference))
data_train_3 = data_preprocessing('eeg1.edf', (active, reference))

data_test = data_preprocessing('eeg1.edf', (active, reference))


# 2) Train the NSDs

# 3) Make predictions on the test data 

# 4) Aggregate predictions

# 5) Find seizure events 

# 6) Calculate metrics 

# 7) Plot results 