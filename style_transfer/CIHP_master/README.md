```
find  /sdc/fiinsh/datasets/fashionAI/train1/Images/* -type f > /sdc/fiinsh/datasets/fashionAI/train1/Images/img_list.txt
split -n l/4 /sdc/fiinsh/datasets/fashionAI/train1/Images/img_list.txt
```
## Part Grouping Network (PGN)
Ke Gong, Xiaodan Liang, Yicheng Li, Yimin Chen, Ming Yang and Liang Lin, "Instance-level Human Parsing via Part Grouping Network", ECCV 2018 (Oral).

### Introduction

PGN is a state-of-art deep learning methord for semantic part segmentation, instance-aware edge detection and instance-level human parsing built on top of [Tensorflow](http://www.tensorflow.org).

This distribution provides a publicly available implementation for the key model ingredients reported in our latest [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_Gong_Instance-level_Human_Parsing_ECCV_2018_paper.pdf) which is accepted by ECCV 2018.


### Crowd Instance-level Human Parsing (CIHP) Dataset

The PGN is trained and evaluated on our [CIHP dataset](http://www.sysu-hcp.net/lip) for isntance-level human parsing.  Please check it for more model details. The dataset is also available at [google drive](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M?usp=sharing) and [baidu drive](http://pan.baidu.com/s/1nvqmZBN).

### Pre-trained models

We have released our trained models of PGN on CIHP dataset at [google drive](https://drive.google.com/open?id=1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk).

### Inference
1. Download the pre-trained model and store in $HOME/checkpoint.
2. Prepare the images and store in $HOME/datasets.
3. Run test_pgn.py.
4. The results are saved in $HOME/output

### Training
1. Download the pre-trained model and store in $HOME/checkpoint.
2. Download CIHP dataset or prepare your own data and store in $HOME/datasets.
3. For CIHP dataset, you need to generate the edge labels and left-right flipping labels (optional). We have provided a script for reference.
4. Run train_pgn.py to train PGN.
5. Use test_pgn.py to generate the results with the trained models.
