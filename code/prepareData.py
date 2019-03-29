import random
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d')
parser.add_argument('--gpu', '-g')
parser.add_argument('--ratio', '-r')
args = parser.parse_args()

f = open('../datasets/%s/graph.txt' % args.dataset, 'rb')
edges = [i for i in f]
selected = int(len(edges) * float(args.ratio))
selected = selected - selected % config.batch_size
selected = random.sample(edges, selected)
remain = [i for i in edges if i not in selected]
fw1 = open('temp/graph.txt', 'wb')
fw2 = open('temp/test_graph.txt', 'wb')

for i in selected:
    fw1.write(i)
for i in remain:
    fw2.write(i)
