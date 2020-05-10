import networkx as nx
import math
import random
import statistics 
import sys
import os


class TestLinkPrediction:

	def __init__(self, G):
		self.G_original = G.copy()
		self.n = self.G_original.number_of_nodes()
		self.m = self.G_original.number_of_edges()



	def create_test_links(self):
		G_test = self.G_original.copy()
		nodes = [node for node in self.G_original.nodes]
		edges = [edge for edge in self.G_original.edges]
		test_edges = min(1000, self.m // 100)

		positive = set()
		while (len(positive) < test_edges):
			ind = random.randint(0, self.m - 1)
			if(edges[ind] not in positive):
				positive.add(edges[ind])

		negative = set()
		while (len(negative) < test_edges):
			x = nodes[random.randint(0, self.n - 1)]
			y = nodes[random.randint(0, self.n - 1)]

			if((x, y) not in edges and (y, x) not in edges and
			   (x, y) not in negative and (y, x) not in negative):
				negative.add((x, y))

		positive = list(positive)
		negative = list(negative)

		for edge in positive:
			G_test.remove_edge(*edge)

		return G_test, positive, negative


	
	@staticmethod
	def preferential_attachment(G, x, y):
		degrees = G.degree
		return degrees[x] * degrees[y]

	@staticmethod
	def jaccard(G, x, y):
		neighbors_x = set([node for node in G.neighbors(x)]) 
		
		common = 0.0
		for node in G.neighbors(y):
			if(node in neighbors_x):
				common += 1

		if(len(G[x]) + len(G[y]) == 0):
			return 1.0

		val = 1.0 * common / (len(G[x]) + len(G[y]) - common)
		return val


	@staticmethod
	def adamic_adar(G, x, y):
		degrees = G.degree
		neighbors_x = set([node for node in G.neighbors(x)]) 
		
		val = 0.0
		for node in G.neighbors(y):
			if(node in neighbors_x):
				val += math.log2(degrees[node])

		return val


	def calculate_similarity(self, similarity_function, G_test, positive, negative):
		positive_weights = []
		for x, y in positive:
			positive_weights.append(similarity_function(G_test, x, y))

		negative_weights = []
		for x, y in negative:
			negative_weights.append(similarity_function(G_test, x, y))

		return positive_weights, negative_weights



	def evaluate_AUC(self, positive_weights, negative_weights):
		test_edges = min(1000, self.m // 100)
		correct = 0
		equal = 0

		for it in range(test_edges):
			x = random.randint(0, len(positive_weights) - 1)
			y = random.randint(0, len(negative_weights) - 1)
			x = positive_weights[x]
			y = negative_weights[y]


			if(abs(x - y) < 10**(-7)):
				equal += 1
			elif x > y:
				correct += 1

		return (correct + equal / 2) / test_edges



	def test(self, iterations=1):
		AUC_preferential = 0.0
		AUC_jaccard = 0.0
		AUC_adamic_adar = 0.0

		for it in range(iterations):
			print("iteration:" , it)
			G_test, positive, negative = self.create_test_links()

			print("generation done")

			positive_weights, negative_weights = self.calculate_similarity(TestLinkPrediction.preferential_attachment, G_test, positive, negative)
			AUC_preferential += self.evaluate_AUC(positive_weights, negative_weights)

			print("preferential done")

			positive_weights, negative_weights = self.calculate_similarity(TestLinkPrediction.jaccard, G_test, positive, negative)
			AUC_jaccard += self.evaluate_AUC(positive_weights, negative_weights)

			print("jaccard done")

			positive_weights, negative_weights = self.calculate_similarity(TestLinkPrediction.adamic_adar, G_test, positive, negative)
			AUC_adamic_adar += self.evaluate_AUC(positive_weights, negative_weights)

			print("adamic_adar done")


		AUC_preferential /= iterations
		AUC_jaccard /= iterations
		AUC_adamic_adar /= iterations

		print("AUC preferential:", AUC_preferential)
		print("AUC Jaccard:", AUC_jaccard)
		print("AUC Adamic-Adar:", AUC_adamic_adar)

		return (AUC_preferential, AUC_jaccard, AUC_adamic_adar)



def AUC_statistics(AUCs, name):
	print(name)
	print("Min AUC:", min(AUCs))
	print("Max AUC:", max(AUCs))
	print("Average AUC:", 1.0 * sum(AUCs) / len(AUCs))
	print("Standard deviation AUC:", statistics.stdev(AUCs))
	print("\n")




AUCs_preferential = []
AUCs_jaccard = []
AUCs_adamic_adar = []


path = "./data/facebook100/"
file_names = os.listdir(path)
for file_name in file_names:
	file_path = path + "/" + file_name + "/" + file_name + ".net"
	G = nx.read_pajek(file_path)
	G = nx.Graph(nx.to_undirected(G))

	print("File:", file_name)
	T = TestLinkPrediction(G)
	AUC_preferential, AUC_jaccard, AUC_adamic_adar = T.test()
	
	AUCs_preferential.append(AUC_preferential)
	AUCs_jaccard.append(AUC_jaccard)
	AUCs_adamic_adar.append(AUC_adamic_adar)


AUC_statistics(AUCs_preferential, "Preferential")
AUC_statistics(AUCs_jaccard, "Jaccard")
AUC_statistics(AUCs_adamic_adar, "Adamic-Adar")