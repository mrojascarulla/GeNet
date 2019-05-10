# GeNet

Genet is a convolutional model for shotgun metagenomic classification. Code is provided for training GeNet using the dataset from [1]. 

## Quick guide

### Install required dependencies

To install the required Python packages, install [Anaconda](https://www.anaconda.com/) and create a virtual environment by executing:

```
bash install_requirements.sh
```

Extact the NCBI taxonomy:
```
cd data
tar -xvzf nodes.dmp.tar.gz
rm nodes.dmp.tar.gz
```

### Train the model

To train the model, run the following:

```
source activate genet
cd code
python genet_train.py --dir_path=../data
```

The first execution of the code will download the required genomes from NCBI. 


The resulting weights and tensorboard log will be saved in GeNet/saved_models.

### References

[*[1] Rojas-Carulla, M., Tolstikhin, I., Luque, G., Youngblut, N., Ley, R. and Schölkopf, B. GeNet: Deep Representations for Metagenomics. arXiv e-print: 1901.11015. 2019. *](https://arxiv.org/abs/1901.11015)
