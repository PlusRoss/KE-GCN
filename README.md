# GEM-GCN

This repository is the official implementation of [Generalized Multi-Relational Graph Convolution Network](https://arxiv.org/abs/2006.07331) in The Web Conference (WWW) 2021. We follow the code style of [tkipf/gcn](https://github.com/tkipf/gcn).

## Requirements

python >= 3.6.0      
tensorflow = 1.15.0    
scipy = 1.4.1    

## Datasets

The initial datasets of knowledge graph alignment task can be found in [JAPE](https://github.com/nju-websoft/JAPE).   
The AM dataset can be found in [RGCN](https://github.com/tkipf/relational-gcn).   
The wordnet dataset can be found in [DHNE](https://github.com/tadpole/DHNE).   
The FB15k dataset can be found in [DKRL](https://github.com/xrb92/DKRL).

## Reproduce Results

### 1. Entity Alignment

To train models for entity alignment task, run these commands:
```
./run_align.sh 0 zh_en QuatE --save
./run_align.sh 0 ja_en QuatE --save
./run_align.sh 0 fr_en QuatE --save
```
where 0 is the GPU index. zh_en is the name of dataset. QuatE is the name of knowledge graph completion method incorporated by our model,
feel free to replace it with: RotatE, TransE, TransD, TransH, DistMult.   

### 2. Relation Alignment

To train models for relation alignment task, run these commands:
```
./run_rel_align.sh 0 zh_en TransE --save
./run_rel_align.sh 0 ja_en TransE --save
./run_rel_align.sh 0 fr_en TransE --save
```

### 3. Entity Classification

To train models for entity classification task, run these commands:   
```
./run_class.sh 0 wordnet TransE --save
./run_class.sh 0 fb15k TransE --save
./run_class.sh 0 am TransE --save
```  
For AM dataset, you need to generate *am.pickle* following [RGCN](https://github.com/tkipf/relational-gcn) and place it into */data/class*. However, we do not recommend using this dataset because its number of labels is very limited.
