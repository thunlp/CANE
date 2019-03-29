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

    python3 run.py --dataset [cora,HepTh,zhihu] --gpu gpu_id --ratio [0.15,0.25,...] --rho rho_value
    
For example, you can train like:

    python3 run.py --dataset zhihu --gpu 0 --ratio 0.55 --rho 1.0,0.3,0.3


## Dependencies
* Tensorflow == 1.11.0
* Scipy == 1.1.0
* Numpy == 1.16.2

## Cite
If you use the code, please cite this paper:
  
_Cunchao Tu, Han Liu, Zhiyuan Liu, Maosong Sun. CANE: Context-Aware Network  Embedding for Relation Modeling. The 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017)._

For more related works on network representation learning, please refer to my [homepage](http://thunlp.org/~tcc/).