
import sklearn.linear_model


def train_lr(training_data):

	x, target = training_data

	lr = sklearn.linear_model.LogisticRegression()
	lr.fit(x, target)

	return lr 


def predict_lr(model, data):

	return model.predict_proba(data)[:, 1]
