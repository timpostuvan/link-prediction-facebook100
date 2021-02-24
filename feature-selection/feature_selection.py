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


#######################################################################################################

	# Rename features as requested by the reviewer
	rename_cols = {"is_dorm": "same_dorm", "is_year": "same_year", "from_high_school": "high_school_1",
			  "to_high_school": "high_school_2", "from_major": "major_1", "to_major": "major_2",
			  "is_faculty": "same_faculty", "is_gender": "same_gender"}

	data.rename(columns=rename_cols, inplace=True)
#######################################################################################################
	

	# Shuffle data
	data = shuffle(data)
	train_data = data

	features = data.columns[:-1].to_numpy()

	X_train = data.iloc[:, :-1].to_numpy()
	X_train = scaler.fit_transform(X_train)
	y_train = data["label"].to_numpy()


	# Read and rescale test data
	data = pd.read_table("../data/" + file_name + "_test.data")

	X_test = data.iloc[:, :-1].to_numpy()
	X_test = scaler.fit_transform(X_test)
	y_test = data["label"].to_numpy()

	return X_train, X_test, y_train, y_test, features, train_data



def write_data(file_name, columns):
	columns = np.append(columns, ["label"])

	data = pd.read_table("../data/" + file_name + "_train.data")
	filtered_data = data[columns]
	filtered_data.to_csv("../data/" + file_name + "_train_filtered.data", sep="\t", index=False)

	data = pd.read_table("../data/" + file_name + "_test.data")
	filtered_data = data[columns]
	filtered_data.to_csv("../data/" + file_name + "_test_filtered.data", sep="\t", index=False)

	data = pd.read_table("../unseen-data/" + file_name + "_unseen.data")
	filtered_data = data[columns]
	filtered_data.to_csv("../unseen-data/" + file_name + "_unseen_filtered.data", sep="\t", index=False)




#X_train, X_test, y_train, y_test, features, data = read_data("baseline")
X_train, X_test, y_train, y_test, features, data = read_data("topological")
#X_train, X_test, y_train, y_test, features, data = read_data("node2vec")
#X_train, X_test, y_train, y_test, features, data = read_data("node_based")



print("Data read")

SVM_classifier = SVC(kernel="linear")

rfecv = RFECV(estimator=SVM_classifier, step=1, cv=StratifiedKFold(5),
              scoring='accuracy', n_jobs=-1,verbose=True)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)


selected_features = np.array([features[i] for i in range(len(features)) if rfecv.support_[i]])
print(selected_features)



<<<<<<< HEAD
=======

>>>>>>> 7cb1ea0ad7b9deae4a0b74b7f10f2f1b7dcadf3c
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


plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices], size=21, wrap=True)
plt.xlabel('Relative Importance', size=21)
plt.show()




# Plot correlation heatmap
correlation_matrix = data.corr()
top_correlation_features = correlation_matrix.index

plt.rcParams['font.size'] = 16
sns.heatmap(data[top_correlation_features].corr().round(2), annot=True, fmt=".2", cmap="RdYlGn")

plt.ylim(15, 0)
plt.xlim(0, 15)
plt.show()