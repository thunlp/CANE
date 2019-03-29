import random

f = open('../datasets/zhihu/graph.txt', 'rb')
edges = [i for i in f]
selected = random.sample(edges, int(len(edges) * 0.95))
remain = [i for i in edges if i not in selected]
fw1 = open('graph.txt', 'wb')
fw2 = open('test_graph.txt', 'wb')

for i in selected:
    fw1.write(i)
for i in remain:
    fw2.write(i)
