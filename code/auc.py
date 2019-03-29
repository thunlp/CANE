import random
import numpy as np
import os

node2vec = {}
dataset_name = "zhihu"
f = open('embed.txt', 'rb')
for i, j in enumerate(f):
    if j.decode() != '\n':
        node2vec[i] = list(map(float, j.strip().decode().split(' ')))
f1 = open(os.path.join('test_graph.txt'), 'rb')
edges = [list(map(int, i.strip().decode().split('\t'))) for i in f1]
nodes = list(set([i for j in edges for i in j]))
a = 0
b = 0
for i, j in edges:
    if i in node2vec.keys() and j in node2vec.keys():
        dot1 = np.dot(node2vec[i], node2vec[j])
        random_node = random.sample(nodes, 1)[0]
        while random_node == j or random_node not in node2vec.keys():
            random_node = random.sample(nodes, 1)[0]
        dot2 = np.dot(node2vec[i], node2vec[random_node])
        if dot1 > dot2:
            a += 1
        elif dot1 == dot2:
            a += 0.5
        b += 1

print(float(a) / b)
