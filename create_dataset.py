import os
import networkx as nx
import math
import random


feature_names = ["from_id", "to_id", "is_dorm", "is_year", "year_diff", "from_high_school",
				 "to_high_school", "from_major", "to_major", "is_faculty", "is_gender", "label"]

node_attributes = ["student_fac", "gender", "major_index", "second_major", "dorm", "year", "high_school"]

attribute_dict = {
    "student_fac" : 0,
    "gender" : 1,
    "major_index" : 2,
    "second_major" : 3,
    "dorm" : 4,
    "year" : 5,
    "high_school" : 6,
    }


#---------------------------------------------------------------------------

class SimilartyScores:
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


#----------------------------------------------------------------------------


def output_feature_set(file_path, data_features):
	f = open(file_path + ".raw_data", "w+")
	f.write("\t".join(feature_names) + "\n")
	for instance in data_features:
		str_instance = list(map(str, instance))
		f.write("\t".join(str_instance) + "\n")
	f.close()



def output_topological_feature_set(file_path, data_topological_features):
	f = open(file_path + ".data", "w+")
	f.write("\t".join(["preferential", "jaccard", "adamic adar", "resource allocation", *(feature_names[2:])]) + "\n")
	for instance in data_topological_features:
		str_instance = list(map(str, instance))
		f.write("\t".join(str_instance) + "\n")
	f.close()



def read_graph(file_path):
	# Read graph from .net file
	G = nx.read_pajek(file_path + ".net")
	G = nx.Graph(nx.to_undirected(G))

	G = nx.convert_node_labels_to_integers(G)


	# Read attibutes from .nodelist file
	f = open(file_path + ".nodelist", "r")
	f.readline()
	for line in f.read().split("\n"):
		if(line == ""):
			break

		attributes = list(map(int, line.split("\t")))
		node_id = attributes[0]
		attributes = attributes[1:]

		for key in attribute_dict:
			index = attribute_dict[key]
			G.nodes[node_id][index] = attributes[index]

	f.close()
	return G



def create_features(G, data):
	# Calculate mean year
	sampled_nodes = []
	for x, y, z in data:
		sampled_nodes.append(x)
		sampled_nodes.append(y)


	mean_year = 0.0
	num_years = 0
	for x in sampled_nodes:
		if(G.nodes[x][attribute_dict["year"]] != 0):
			mean_year += G.nodes[x][attribute_dict["year"]]
			num_years += 1

	mean_year /= num_years

	# Complete missing year data with mean year
	for x in sampled_nodes:
		if(G.nodes[x][attribute_dict["year"]] == 0):
			G.nodes[x][attribute_dict["year"]] = mean_year

	data_features = []
	for x, y, l in data:
		# from_id, to_id, label
		from_id, to_id, label = x, y, l

		#is_dorm
		dorms = [G.nodes[x][attribute_dict["dorm"]], G.nodes[y][attribute_dict["dorm"]]]
		is_dorm = int(dorms[0] == dorms[1])
		if(min(dorms) == 0):
			is_dorm = 0


		#is_year, year_diff
		years = [G.nodes[x][attribute_dict["year"]], G.nodes[y][attribute_dict["year"]]]
		is_year = int(years[0] == years[1])
		year_diff = abs(years[0] - years[1])


		# from_high_school, to_high_school
		high_schools = [G.nodes[x][attribute_dict["high_school"]], G.nodes[y][attribute_dict["high_school"]]]
		from_high_school = min(high_schools)
		to_high_school = max(high_schools)


		#from_major, to_major
		majors = [G.nodes[x][attribute_dict["major_index"]], G.nodes[y][attribute_dict["major_index"]]]
		from_major = min(majors)
		to_major = max(majors)


		#is_faculty
		faculties = [G.nodes[x][attribute_dict["student_fac"]], G.nodes[y][attribute_dict["student_fac"]]]
		is_faculty = int(faculties[0] == faculties[1])
		if(min(faculties) == 0):
			is_faculty = 0


		#is_gender
		genders = [G.nodes[x][attribute_dict["gender"]], G.nodes[y][attribute_dict["gender"]]]
		is_gender = int(genders[0] == genders[1])
		if(min(genders) == 0):
			is_gender = 0


		features = [from_id, to_id, is_dorm, is_year, year_diff, from_high_school,
				 	to_high_school, from_major, to_major, is_faculty, is_gender, label]
		data_features.append(features)

	return data_features



def create_topological_features(G, data_features):
	data_topological_features = []
	for instance in data_features:
		x, y = instance[:2]

		preferential = SimilartyScores.preferential_attachment(G, x, y)
		jaccard = SimilartyScores.jaccard(G, x, y)
		adamic_adar = SimilartyScores.adamic_adar(G, x, y)
		resource_allocation = SimilartyScores.resource_allocation(G, x, y)

		data_topological_features.append([preferential, jaccard, adamic_adar, resource_allocation, *instance[2:]])

	return data_topological_features



def create_test_links(G, percentage):
	n = G.number_of_nodes()
	m = G.number_of_edges()
	nodes = [node for node in G.nodes]
	edges = [edge for edge in G.edges]
	test_edges = int (m * percentage)

	positive = set()
	while (len(positive) < test_edges):
		ind = random.randint(0, m - 1)
		if(edges[ind] not in positive):
			positive.add(edges[ind])

	negative = set()
	while (len(negative) < test_edges):
		x = nodes[random.randint(0, n - 1)]
		y = nodes[random.randint(0, n - 1)]

		if((x, y) not in edges and (y, x) not in edges and
		   (x, y) not in negative and (y, x) not in negative):
			negative.add((x, y))

	positive_labeled = [(x, y, 1) for x, y in positive]
	negative_labeled = [(x, y, 0) for x, y in negative]

	return positive_labeled, negative_labeled





path = "./data/"
file_names = os.listdir(path)


for file_name in file_names:
	file_path = path + "/" + file_name + "/" + file_name
	print("File:", file_name)

	G = read_graph(file_path)
	positive_labeled, negative_labeled = create_test_links(G, 0.02)
	data = positive_labeled
	data.extend(negative_labeled)

	data_features = create_features(G, data)
	data_topological_features = create_topological_features(G, data_features)

	output_feature_set(file_path, data_features)
	output_topological_feature_set(file_path, data_topological_features)