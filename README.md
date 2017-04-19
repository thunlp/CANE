# CANE
Source code and datasets of ACL2017 paper: "CANE: Context-Aware Network  Embedding for Relation Modeling"

## Datasets
This folder "datasets" contains three datasets used in CANE, including Cora, HepTh and Zhihu. In each dataset, there are two files named "data.txt" and "graph.txt".

* data.txt: Each line represents the text information of a vertex.    
* graph.txt: The edgelist file of current social network.

Besides, there is an additional "group.txt" file in Cora.

* group.txt: Each vertex in Cora has been annotated with a label. This file can be used for vertex classification.

## Run
    Run the following command for training CANE:
		python train.py

## Dependencies
	*	Tensorflow == 0.12
	*	Scipy == 0.18.1
	*	Numpy == 1.11.2

## Cite
  If you use the code, please cite this paper:
  
  Cunchao Tu, Han Liu, Zhiyuan Liu, Maosong Sun. CANE: Context-Aware Network  Embedding for Relation Modeling. The 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017).