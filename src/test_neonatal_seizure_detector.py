
import torch 
import scipy.special


def predict(device, model, data):
	
	model.eval()
	model.to(device)

	(n_batch, _, _) = data.shape

	p = np.zeros(n_batch)

	mini_batch = 128

	for t in range(0, n_batch, mini_batch):
		t_end = min(t + mini_batch, n_batch)

		tmp_data = data[t:t_end, :, :].to(device)
		score = model.forward(tmp_data).detach().cpu().numpy()
		probabilities = scipy.special.softmax(score, axis = 1)

		p[t:t_end] = probabilities[:, 1]

	return p


def example_run():

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
	model = torch.load('test_model.pt', map_location = torch.device('cpu')) 
	X = torch.rand(20, 18, 512)

	p = predict(device, model, X)
	print(p)