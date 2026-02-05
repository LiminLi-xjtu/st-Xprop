# st-Xprop

**st-Xprop** is a method for spatial domains identification in multi-modal spatial transcriptomics. This repository contains code and examples for running st-Xprop on real spatial transcriptomics datasets.

![image](https://github.com/LiminLi-xjtu/st-Xprop/blob/master/st-Xprop_arch.jpg)

## System Requirements

st-Xprop is implemented in **Python 3.9.18**. All results presented in the paper were obtained using an **NVIDIA GeForce RTX 4090** GPU.


## Requirements

To run st-Xprop, you need to install the following dependencies:

- `h5py==3.9.0`
- `matplotlib==3.10.8`
- `numpy==2.4.2`
- `pandas==3.0.0`
- `Pillow==9.4.0`
- `Pillow==12.1.0`
- `POT==0.9.5`
- `rpy2==3.6.2`
- `rpy2_rinterface==3.6.2`
- `rpy2_robjects==3.6.1`
- `scanpy==1.12`
- `scikit_learn==1.8.0`
- `scipy==1.17.0`
- `timm==1.0.24`
- `torch==2.4.0`
- `torchvision==0.19.0`
- `tqdm==4.67.1`
- `R==4.5.1`


## Example Usage

The input data for st-Xprop should be in the .h5ad or .h5 format. If your data is in a different format, refer to the [Scanpy](https://scanpy.readthedocs.io/en/stable/) or [anndata](https://anndata.readthedocs.io/en/stable/) tutorials for instructions on how to convert your data into the .h5ad format.

For histological data, the preprocessing method can be either stMVC (Preprocessing_stMVC.py) or a pre-trained Vision Transformer (Preprocessing_ViT.py)


### Data access

All datasets used in this study have been curated and deposited at [Zenodo](https://zenodo.org/records/18449464). 

### Configuration

The configuration file contains the parameter settings and data information for different datasets. You can find it in the `config/` folder. The settings include important hyperparameters such as learning rates, batch sizes, and specific model parameters tailored for each dataset, as well as the paths to the input datasets. You can modify these settings according to your requirements.

Example parameter settings in the config file:

```bash
# Dataset and transformation settings
dataset_name: BMMC_paired  # Name of the dataset (e.g., BMMC (paired)). Specify which dataset you are working with.
dataset_dir: ../data/BMMC_paired  # Path to the BMMC (paired) data directory. This directory should include files like rna.h5ad, atac.h5ad, and scGAM_ArchR.h5ad.
                          # These datasets can be downloaded from [Zenodo](https://zenodo.org/uploads/14506611).
GAM_name: ArchR  # Transformation method used to convert the ATAC-seq data (atac.h5ad) into a gene activity score matrix.
                 # Options could include 'ArchR' or 'Signac', depending on your preprocessing method.
dataset_type: RNA_ATAC  # Type of data integration. This defines which modalities you are integrating. 
                       # Options could be:
                       # - 'RNA_ATAC' for RNA and ATAC-seq integration
                       # - 'RNA_Protein' for RNA and Protein integration
batch: None  # Batch information for BMCITE (CITE-seq) data. If no batch information is available, set this parameter to None.
paired: True  # Whether the data is paired (True/False). 'True' if the data are from paired modalities (e.g., scRNA-seq and scATAC-seq), 
              # 'False' otherwise (e.g., unpaired datasets).

# Model hyperparameters
n_high_var: 2000  # The number of highly variable genes (HVGs) selected for preprocessing. 
dim: 100  # The dimensionality of PCA embeddings in the preprocessing step. It reduces the number of features while retaining variance.
neighbors_mnn: 500  # The number of nearest neighbors used for Mutual Nearest Neighbor (MNN) construction. 
metric: cosine  # The distance metric to use for MNN construction. 
                # You can also choose other distance metrics like 'euclidean' depending on the nature of your data.
use_rep: hvg_count  # The representation used for gene features. Options include:
                   # - 'hvg_count' for using raw counts of high-variance genes
                   # - 'hvg_norm' for using normalized data
                   # - 'low_emb' for using low-dimensional embeddings from previous steps as feature representations.

latent_dim: 50  # The dimensionality of the latent space for the model. 

# Regularization parameters
tau_cell: 0.5  # Hyperparameter for the cell-level contrastive loss function, default is 0.5. 
tau_feature: 0.5  # Hyperparameter for the feature-level contrastive loss function, default is 0.5.
gamma_a: 1 # The weight for modality 1 reconstruction regularization, default is 1.
gamma_b: 1 # The weight for modality 2 reconstruction regularization, default is 1.
alpha: 10000  # The weight for cell-level regularization, default is 10000. 
beta: 10000  # The weight for feature-level regularization, default is 10000. 

# Training settings
seed: 123  # The random seed for reproducibility
batch_size: 256  # The batch size for training
learning_rate: 0.0001  # The learning rate for optimization, default is 0.0001
weight_decay: 0.00005  # The weight decay (L2 regularization) applied to model parameters to prevent overfitting and encourage simpler models, deafult is 0.00005.
epoch: 1000  # The number of training epochs

```

### Tutorial

Please find examples of st-Xprop applications in the tutorial folder, where jupyter notebooks are provided.
