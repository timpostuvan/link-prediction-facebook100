import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
import sklearn


class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()

		self.d1 = nn.Linear(3, 10)
		self.d2 = nn.Linear(10, 1)

		
	def forward(self, x):
		x = self.d1(x)
		x = F.relu(x)

		x = self.d2(x)
		out = torch.sigmoid(x)
		return out



class FacebookDataset(Dataset):
	def __init__(self, file_path):
		data = pd.read_table(file_path)
		self.data =  data.iloc[:, :-1].to_numpy().astype(np.float32)
		self.labels = data["label"].to_numpy().astype(np.float32)

		scaler = StandardScaler()
		self.data = scaler.fit_transform(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]




torch.multiprocessing.set_sharing_strategy('file_system')
train_data = FacebookDataset("../data/baseline_train_filtered.data")
train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=2)

test_data = FacebookDataset("../data/baseline_test_filtered.data")
test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=2)

unseen_data = FacebookDataset("../unseen-data/baseline_unseen_filtered.data")
unseen_dataloader = DataLoader(unseen_data, batch_size=50, shuffle=False, num_workers=2)



learning_rate = 0.001
num_epochs = 5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork()
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
	train_running_loss = 0.0
	train_acc = 0.0
	all_labels = []
	all_predict = []

	## training step
	for i, (x, labels) in enumerate(train_dataloader):
		x = x.to(device)
		labels = labels.to(device)

		## forward + backprop + loss
		logits = model(x)
		loss = criterion(logits, labels.unsqueeze(1))
		optimizer.zero_grad()
		loss.backward()

		## update model params
		optimizer.step()

		train_running_loss += loss.detach().item()

		predict = torch.round(logits)
		train_acc += (predict == labels.unsqueeze(1)).type(torch.float).mean().float()

		all_labels.extend(list(labels))
		all_predict.extend(list(logits))


	all_predict = np.array(all_predict)
	all_labels = np.array(all_labels)

	print("AUC ROC:", sklearn.metrics.roc_auc_score(all_labels, all_predict))
	print('Epoch: %d | Loss: %.4f | Train Accuracy: %.4f' \
	  %(epoch, train_running_loss / i, train_acc/i))



## Test data

test_acc = 0.0
test_labels = []
test_predict = []
for i, (x, labels) in enumerate(test_dataloader):
	x = x.to(device)
	labels = labels.to(device)
	outputs = model(x)
	
	predict = torch.round(outputs)
	test_acc += (predict == labels.unsqueeze(1)).type(torch.float).mean().float()

	test_labels.extend(list(labels))
	test_predict.extend(list(outputs))


test_predict = np.array(test_predict)
test_labels = np.array(test_labels)

print("TEST DATA")
print("Test AUC ROC:", sklearn.metrics.roc_auc_score(test_labels, test_predict))
print('Test Accuracy: %.4f'%(test_acc/i))



## Unseen data

unseen_acc = 0.0
unseen_labels = []
unseen_predict = []
for i, (x, labels) in enumerate(unseen_dataloader):
	x = x.to(device)
	labels = labels.to(device)
	outputs = model(x)
	
	predict = torch.round(outputs)
	unseen_acc += (predict == labels.unsqueeze(1)).type(torch.float).mean().float()

	unseen_labels.extend(list(labels))
	unseen_predict.extend(list(outputs))


unseen_predict = np.array(unseen_predict)
unseen_labels = np.array(unseen_labels)

print("UNSEEN DATA")
print("Unseen AUC ROC:", sklearn.metrics.roc_auc_score(unseen_labels, unseen_predict))
print('Unseen Accuracy: %.4f'%(unseen_acc/i))