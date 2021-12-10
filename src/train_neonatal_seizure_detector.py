
import torch
import numpy as np
import itertools

from neonatal_seizure_detector import NSD 


def train(device, training_data, learning_rate = 0.001, step_size = 10, gamma = 0.5, epochs_n = 30):

	model = NSD()
	model.to(device) # send the model to the specified device
	model.train()

	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma) 
	loss_function = torch.nn.CrossEntropyLoss() 

	for _ in range(epochs_n):

		for data, labels in training_data: 

			optimizer.zero_grad()
		
			data, labels = data.float().to(device), labels.long().to(device)  
			pred = model.forward(data)
			loss = loss_function(pred, labels)

			loss.backward()
			optimizer.step()

		scheduler.step()

	return model


def example_run(): 

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

	n_batch, batch_size, n_channels = 3, 32, 18
	X, y = torch.rand(n_batch, batch_size, n_channels, 512), torch.randint(2, (n_batch, batch_size))

	m = train(device, zip(X, y))
	torch.save(m, 'test_model.pt') 

