
import torch


class AttentionLayer(torch.nn.Module):
	"""
	D. Y. Isaev, D. Tchapyjnikov, C. M. Cotten, D. Tanaka, N. Martinez,
	M. Bertran, G. Sapiro, and D. Carlson, “Attention-based network for
	weak labels in neonatal seizure detection,” Proceedings of machine
	learning research, vol. 126, p. 479, 2020.
	"""

	def __init__(self, size_in, size_inner):
		super().__init__()

		self.size_in, self.size_innter = size_in, size_inner
		self.w = torch.nn.Parameter(torch.rand(1, size_inner, 1))  
		self.V = torch.nn.Parameter(torch.rand(1, size_inner, size_in))

	def attention_coefficients(self, x):
		(_, n_channels, _) = x.shape
		s1 = torch.transpose(x, 1, 2)
		s2 = torch.matmul(self.V, s1)
		s3 = torch.tanh(s2)
		s4 = torch.transpose(self.w, 1, 2)
		s5 = torch.matmul(s4, s3)
		s6 = torch.exp(s5)
		s7 = torch.sum(s6, 2)
		s7 = s7.repeat(1, n_channels).view(-1, 1, n_channels)
		a = torch.div(s6, s7)
		return a

	def forward(self, x):
		a = self.attention_coefficients(x)
		return torch.sum(torch.transpose(a, 1, 2) * x, 1)


class NSD(torch.nn.Module):
	"""
	A. O’Shea, G. Lightbody, G. Boylan, and A. Temko, “Investigating the
	impact of CNN depth on neonatal seizure detection performance,” in
	2018 40th Annual International Conference of the IEEE Engineering in
	Medicine and Biology Society (EMBC), pp. 5862–5865, IEEE, 2018
	"""

	def __init__(self):

		super().__init__()

		self.relu = torch.nn.ReLU()
	
		self.cnn1 = torch.nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 3)
		self.batch1 = torch.nn.BatchNorm1d(num_features = 32)
		 
		self.cnn2 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch2 = torch.nn.BatchNorm1d(num_features = 32)
		
		self.cnn3 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch3 = torch.nn.BatchNorm1d(num_features = 32) 
			 
		self.avgpool1 = torch.nn.AvgPool1d(kernel_size = 8, stride = 3)
		self.cnn4 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch4 = torch.nn.BatchNorm1d(num_features = 32)

		self.cnn5 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch5 = torch.nn.BatchNorm1d(num_features = 32)
	
		self.cnn6 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch6 = torch.nn.BatchNorm1d(num_features = 32)
	
		self.avgpool2 = torch.nn.AvgPool1d(kernel_size = 4, stride = 3)
		self.cnn7 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch7 = torch.nn.BatchNorm1d(num_features = 32)
		
		self.cnn8 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch8 = torch.nn.BatchNorm1d(num_features = 32)
		
		self.cnn9 = torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch9 = torch.nn.BatchNorm1d(num_features = 32)
		
		self.avgpool3 = torch.nn.AvgPool1d(kernel_size = 2, stride = 3)
		self.cnn10 =  torch.nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.batch10 = torch.nn.BatchNorm1d(num_features = 32)
			
		self.cnn11 =  torch.nn.Conv1d(in_channels = 32, out_channels = 2, kernel_size = 3)
		self.batch11 = torch.nn.BatchNorm1d(num_features = 2)
			
		self.attention = AttentionLayer(size_in = 24, size_inner = 16)
		self.linear = torch.nn.Linear(in_features = 24, out_features = 2, bias = True)		


	def feature_extractor(self, x):

		(n_batch, n_channels, n_features) = x.shape
		x = x.view(-1, 1, n_features)

		out = self.cnn1(x)
		out = self.batch1(out)
		out = self.relu(out)

		out = self.cnn2(out)
		out = self.batch2(out)
		out = self.relu(out)
		
		out = self.cnn3(out)
		out = self.batch3(out)
		out = self.relu(out)
		
		out = self.avgpool1(out)
		out = self.cnn4(out)
		out = self.batch4(out)
		out = self.relu(out)
		
		out = self.cnn5(out)
		out = self.batch5(out)
		out = self.relu(out)
		
		out = self.cnn6(out)
		out = self.batch6(out)
		out = self.relu(out)
		
		out = self.avgpool2(out)
		out = self.cnn7(out)
		out = self.batch7(out)
		out = self.relu(out)
		
		out = self.cnn8(out)
		out = self.batch8(out)
		out = self.relu(out)
		
		out = self.cnn9(out)
		out = self.batch9(out)
		out = self.relu(out)
		
		out = self.avgpool3(out)
		out = self.cnn10(out)
		out = self.batch10(out)
		out = self.relu(out)

		out = self.cnn11(out)
		out = self.batch11(out)
		out = self.relu(out)

		return out.view(n_batch, n_channels, 24)


	def forward(self, x):

		out = self.feature_extractor(x)
		out = self.attention(out)
		out = self.linear(out)
		# softmax is left out since CrossEntropyLoss expects input without it

		return out












