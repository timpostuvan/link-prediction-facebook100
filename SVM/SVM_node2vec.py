import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def show_statistics(y_test, y_pred, y_values):
	print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))
	print("Test AUC ROC:", sklearn.metrics.roc_auc_score(y_test, y_values))



def read_data(file_name):
	scaler = StandardScaler()
	
	# Read and rescale train data	
	data = pd.read_table("../data/" + file_name + "_train_filtered.data")
	# Shuffle data
	data = shuffle(data)

	X_train = data.iloc[:, :-1].to_numpy()
	X_train = scaler.fit_transform(X_train)
	y_train = data["label"].to_numpy()


	# Read and rescale test data
	data = pd.read_table("../data/" + file_name + "_test_filtered.data")

	X_test = data.iloc[:, :-1].to_numpy()
	X_test = scaler.fit_transform(X_test)
	y_test = data["label"].to_numpy()


	# Read and rescale unseen data
	data = pd.read_table("../unseen-data/" + file_name + "_unseen_filtered.data")

	X_unseen = data.iloc[:, :-1].to_numpy()
	X_unseen = scaler.fit_transform(X_unseen)
	y_unseen = data["label"].to_numpy()

	return X_train, X_test, y_train, y_test, X_unseen, y_unseen





X_train, X_test, y_train, y_test, X_unseen, y_unseen = read_data("node2vec")

print("Data read")

# Train and test SVM
SVM_classifier = SVC(kernel='rbf')
SVM_classifier.fit(X_train, y_train)



print("TEST DATA:")
y_pred = SVM_classifier.predict(X_test)
y_values = SVM_classifier.decision_function(X_test)
show_statistics(y_test, y_pred, y_values)



print("UNSEEN DATA:")
y_pred = SVM_classifier.predict(X_unseen)
y_values = SVM_classifier.decision_function(X_unseen)
show_statistics(y_unseen, y_pred, y_values)
