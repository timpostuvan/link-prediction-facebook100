import os
import numpy as np
import pandas as pd


def partition_dataset(data, train_percentage):
	train_size = int (len(data) * train_percentage) 
	np.random.shuffle(data)

	return data[:train_size], data[train_size:]



if(__name__ == "__main__"):
	path = "./data/"
	file_names = os.listdir(path)

	columns = None
	all_data = None
	exists = False
	for file_name in file_names:
		file_path = path + "/" + file_name + "/" + file_name
		print("File:", file_name)

		curret_data = pd.read_table(file_path + ".data")
		columns = curret_data.columns
		
		if(not exists):
			exists = True
			all_data = curret_data.to_numpy()
		else:
			all_data = np.vstack((all_data, curret_data.to_numpy()))


	train_data, test_data = partition_dataset(all_data, 0.8)
	topological_train_data = pd.DataFrame(train_data, columns=columns)	
	topological_test_data = pd.DataFrame(test_data, columns=columns)

	baseline_train_data = topological_train_data.iloc[:, [0, 1, 2, 3, -1]]
	baseline_test_data = topological_test_data.iloc[:, [0, 1, 2, 3, -1]]

	topological_train_data.to_csv("./data/topological_train.data", sep="\t", index=False)
	topological_test_data.to_csv("./data/topological_test.data", sep="\t", index=False)

	baseline_train_data.to_csv("./data/baseline_train.data", sep="\t", index=False)
	baseline_test_data.to_csv("./data/baseline_test.data", sep="\t", index=False)
