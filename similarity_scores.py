import networkx as nx
import math

class SimilarityScores:
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


	@staticmethod
	def resource_allocation(G, x, y):
		degrees = G.degree
		neighbors_x = set([node for node in G.neighbors(x)]) 
		
		val = 0.0
		for node in G.neighbors(y):
			if(node in neighbors_x):
				val += degrees[node]

		return val