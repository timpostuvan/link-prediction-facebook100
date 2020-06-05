import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier


def read_data(file_name):
	scaler = StandardScaler()
	
	# Read and rescale train data	
	data = pd.read_table("../data/" + file_name + "_train.data")
	# Shuffle data
	data = shuffle(data)
	train_data = data

	features = data.columns[:-1].to_numpy()

	X_train = data.iloc[:10000, :-1].to_numpy()
	X_train = scaler.fit_transform(X_train)
	y_train = data["label"].to_numpy()[:10000]


	# Read and rescale test data
	data = pd.read_table("../data/" + file_name + "_test.data")

	X_test = data.iloc[:10000, :-1].to_numpy()
	X_test = scaler.fit_transform(X_test)
	y_test = data["label"].to_numpy()[:10000]

	return X_train, X_test, y_train, y_test, features, train_data





#X_train, X_test, y_train, y_test, features, data = read_data("baseline")
X_train, X_test, y_train, y_test, features, data = read_data("topological")
#X_train, X_test, y_train, y_test, features, data = read_data("node2vec")


print("Data read")

SVM_classifier = SVC(kernel="linear")

rfecv = RFECV(estimator=SVM_classifier, step=1, cv=StratifiedKFold(5),
              scoring='accuracy', n_jobs=3)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot accuracy 
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




# Decision tree feature importance
decision_tree = ExtraTreesClassifier()
decision_tree.fit(X_train, y_train)

importances = decision_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# Plot correlation heatmap
correlation_matrix = data.corr()
top_correlation_features = correlation_matrix.index

sns.heatmap(data[top_correlation_features].corr(), annot=True, cmap="RdYlGn")

plt.ylim(15, 0)
plt.xlim(0, 15)
plt.show()