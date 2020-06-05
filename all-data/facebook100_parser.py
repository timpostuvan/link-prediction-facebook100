import os, sys, glob
from collections import defaultdict

# Depends on the python library networkx for
# exporting in GML and GraphML.
import networkx as nx

# Depends on scipy to parse matlab files.
import scipy.io

"""
A parser for the Facebook100 dataset.
Data available on https://archive.org/details/oxford-2005-facebook-matrix.

Each of the school .mat files has an A matrix (sparse) and a
"local_info" variable, one row per node: a student/faculty status
flag, gender, major, second major/minor (if applicable), dorm/house,
year, and high school. Missing data is coded 0.

"""

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


def export_graph(G, write_filename):
    write_dir = "./facebook100/" + write_filename + "/"
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)


    print("Writing pajek")
    nx.write_pajek(G, write_dir + write_filename + ".net")

    
    print("Writing nodelist")
    f = open(write_dir + write_filename + ".nodelist", "w")
    f.write("\t".join(["node_id"] + node_attributes) + "\n")
    for node in G.nodes():
        f.write("\t".join([str(node)] + [str(G.nodes[node][attribute]) for attribute in node_attributes]) + "\n")
    
    f.close()



def get_attribute_partition(matlab_object, attribute):
    attribute_rows = matlab_object["local_info"]
    
    try:
        index = attribute_dict[attribute]
    except KeyError:
        raise KeyError("Given attribute " + attribute + " is not a valid choice.\nValid choices include\n" + str(attribute_dict.keys()))

    current_id = 0
    values = dict()
    for row in attribute_rows:
        if not(len(row) == 7):
            raise ValueError("Row " + str(current_id) + " has " + str(len(row)) + " rather than the expected 7 rows!")
       
        val = row[index]
        values[current_id] = int(val)
        current_id += 1

    return values


if __name__ == '__main__':
    if not os.path.isdir("./facebook100/"):
        os.mkdir("./facebook100/")

    matlab_files = glob.glob("./data/*.mat")
    print(onlyfiles)
    matlab_files = filter(lambda fn: not("schools.mat" in fn), matlab_files)
    for matlab_filename in matlab_files:
        network_name = matlab_filename.strip(".").strip("/").split("/")[-1].split(".")[0]
        
        print("Now parsing:", network_name)
        matlab_object = scipy.io.loadmat(matlab_filename)
        scipy_sparse_graph = matlab_object["A"]
        G = nx.from_scipy_sparse_matrix(scipy_sparse_graph)
        
        print("Number of nodes:", G.number_of_nodes())
        print("Number of edges:", G.number_of_edges())
        for attribute in attribute_dict:
            values = get_attribute_partition(matlab_object, attribute)
            for node in values:
                    G.nodes[node][attribute] = values[node]

        export_graph(G, network_name)