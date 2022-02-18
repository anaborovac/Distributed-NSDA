
import torch
import numpy as np

from nsd import NSD 


def prepare_data(x, y, balance = True):

	x_train = x[y != 2]
	y_train = y[y != 2]

	if balance:

		u, c = np.unique(y_train, return_counts = True)
		majority_class = u[np.argmax(c)]

		majority_class_i = np.where(y_train == majority_class)[0]
		majority_class_i = np.random.choice(majority_class_i, size = min(c), replace = False)

		minority_class_i = np.where(y_train != majority_class)[0]
		
		x_train = x_train[np.r_[majority_class_i, minority_class_i]]
		y_train = y_train[np.r_[majority_class_i, minority_class_i]]

	return x_train, y_train


class IterableDataset(torch.utils.data.IterableDataset):

	def __init__(self, data):
		(self.x, self.y) = data


	def __iter__(self):
		p = np.random.permutation(len(self.y))
		for (xx, yy) in zip(self.x[p], self.y[p]):
			yield xx, yy


def collate_fn(batch):

	data = [d for (d, _) in batch]
	labels = [l for (_, l) in batch]

	data = torch.tensor(data).float()
	labels = torch.tensor(labels).long()

	return data, labels


def train(device, training_data, learning_rate = 0.001, step_size = 10, gamma = 0.5, epochs_n = 5, batch_size = 32):

	model = NSD()
	model.to(device) # send the model to the specified device
	model.train()

	iterable_dataset = IterableDataset(training_data) 
	loader = torch.utils.data.DataLoader(iterable_dataset, collate_fn = collate_fn, batch_size = batch_size)

	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma) 
	loss_function = torch.nn.CrossEntropyLoss() 

	print('Start training')

	for _ in range(epochs_n):

		for data, labels in loader: 

			optimizer.zero_grad()
		
			data, labels = data.to(device), labels.to(device)  
			pred = model.forward(data)
			loss = loss_function(pred, labels)

			loss.backward()
			optimizer.step()

		scheduler.step()

	print('Training completed')

	return model


def example_run(): 

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

	n_batch, batch_size, n_channels = 3, 32, 18
	X, y = torch.rand(n_batch, batch_size, n_channels, 512), torch.randint(2, (n_batch, batch_size))

	m = train(device, zip(X, y))
	torch.save(m, 'test_model.pt') 

