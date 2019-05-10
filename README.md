# GeNet

Genet is a convolutional model for shotgun metagenomic classification. Code is provided for training GeNet using the dataset from [1]. 

## Quick guide

### Install required dependencies

To install the required Python packages, install [Anaconda](https://www.anaconda.com/) and create a virtual environment by executing:

```
bash install_requirements.sh
```

Download the NCBI taxonomy by running the following in the main GeNet directory:
```
wget ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
tar -xvzf taxdump.tar.gz
mv nodes.dmp data
rm readme.txt names.dmp merged.dmp gencode.dmp gc.prt division.dmp delnodes.dmp citations.dmp taxdump.tar.gz
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
