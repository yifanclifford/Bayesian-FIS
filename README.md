# Bayesian Feature Interaction Selection

This is the overall implementation of our proposed **BP-FIS** [[1]](#refer-anchor-1) and **BH-FIS** [[2]](#refer-anchor-2). We implement BH-FIS and re-implement BP-FIS. The implementation also contains FMs, HOFMs and DeepFMs. Therefore, you may run our methods and the baselines under the same framework.

<div id="refer-anchor-1">
[1] Yifan Chen, Pengjie Ren, Yang Wang, and Maarten de Rijke. 2019. Bayesian Personalized Feature Interaction Selection for Factorization Machines. SIGIR'19.
</div>
<div id="refer-anchor-2">
[2] Yifan Chen, Yang Wang, Pengjie Ren, Meng Wang and Maarten de Rijke. 2021. Bayesian Feature Interaction Selection for Factorization Machines. The Journal of Artificial Intelligence. (to appear)
</div>


## Requirements

Python 3.x and the required packages:
- argparse
- pyyaml
- pytorch >= 1.7.0
- pytrec_eval

### Install pytrec_eval

We made few changes in pytrec_eval to support mean reciprocal rank cut. Please install pytrec_eval following the instructions below:

- download pytrec_eval from [repo](https://github.com/cvangysel/pytrec_eval/archive/refs/heads/master.zip)
- modify setup.py in pytrec_eval:
```python
REMOTE_TREC_EVAL_URI = 'https://github.com/yifanclifford/trec_eval/archive/refs/heads/mrr_cut.zip'
REMOTE_TREC_EVAL_TLD_NAME = 'trec_eval-mrr_cut'
```
- Install pytrec_eval
```
python setup.py install
```
## Project description
the repo contains four directories:
- FIS: Implementation of Bayesian Feature Interaction Selection. It implements both BP-FIS [[1]](#refer-anchor-1) and BH-FIS [[2]](#refer-anchor-2). It also contains the implementation of relevant baselines.
- process: the code for data processing
- cpp: this is the auxiliary implementations for sampling.
  - Due to the CPU efficiency, it is implemented in C.
  - For the difference of OS, please compile it yourself as follows. It will output a C library in cpp/release. It is required by the python code.
```bash
$ cd cpp
$ chmod +x make.sh
$ ./make.sh
```
- dataset: the datasets used in the paper

and two files:
- readme.md: the description of this repo (**this file**)
- config.yaml: the file provides all configurations of the projects.

### Configuration script
For the ease of using our implementation, we put some parameters settings in the file **config.yaml**. It generally contains the parameter with default values. The parameters can be divided into two categories:

#### Directories and Names
- data_dir: the directory of the dataset (default in ../dataset)
- model_dir: the directory of the saved dataset (default in ../model)
- result_dir: the directory of the output result (default in ../result)
- data_name: the name of the dataset

#### Parameter for training
- batch_size: the batch size of training
- test_size: the batch size of testing
selection_epoch: 1
weight_epoch: 0
selection_lr: 0.01
weight_lr: 0.001
num_candidate: 101


## Data preparation
In our experiments, we use 6 datasets in total. To help reproducibility, we provide scripts of how to prepare each dataset.

### HetRec datasets
You may find the three datasets from HetRec in the [BP-FIS](https://github.com/yifanclifford/BP-FIS) repo:
- [MLHt](https://github.com/yifanclifford/BP-FIS/raw/master/dataset/MLHt.zip)
- [lastFM](https://github.com/yifanclifford/BP-FIS/raw/master/dataset/lastFM.zip)
- [delicious](https://github.com/yifanclifford/BP-FIS/raw/master/dataset/delicious.zip)

Since we re-implement the code for BP-FIS, the format of dataset is slightly different. We provide a script to convert the data format.
```bash
cd process
python hetrec.py
```
Before which, you have to download the datasets, put them into the "../dataset" directory and unzip.

### MovieLens one million
This dataset is shorted as ML1m. It can be fould in /dataset/ML1m.zip.
Inside ML1m found two directories, namely "topn" and "ctr". This is because the ML1m dataset is used for both top-N recommendation and CTR-prediction.

### MovieLens 20 million
Since this is a rather large dataset, we do not upload the processed dataset. Instead, you should download the dataset from the MovieLens website ([ML20m](https://files.grouplens.org/datasets/movielens/ml-20m.zip)). We provide the script the process the dataset.
```bash
cd dataset/ML20m
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
mv ml-20m raw
cd ../../../process
python ml20m.py
```
### Avazu dataset
Similarly, you should download the Avazu dataset and the processing script is provided:
``` bash
python avazu.py
```

## Running example
We consider two tasks: recommendation and CTR-prediction.
### Top-$N$ Recommendation
Here is an example of running an experiment of recommendation:
```Python
python main.py PFIS --task topn --gpu -d 64 --epoch 100 --initial
```
### CTR prediction
