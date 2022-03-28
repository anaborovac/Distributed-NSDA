
import sklearn.linear_model


def train_lr(training_data):
	"""
	Parameters:
		training_data - training data for the logistic regression 

	Output:
		trained/fitted logistic regression model
	"""

	x, target = training_data

	lr = sklearn.linear_model.LogisticRegression()
	lr.fit(x, target)

	return lr 


def predict_lr(model, data):
	"""
	Parameters:
		model - logistic regression model
		data - array of size (N, *) with data to be predicted 

	Output:
		array of size N with predicted probabilities
	"""

	return model.predict_proba(data)[:, 1]
