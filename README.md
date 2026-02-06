# st-Xprop

**st-Xprop** is a method for spatial domain identification in multi-modal spatial transcriptomics data.  
This repository provides the implementation and example workflows for applying st-Xprop to real spatial transcriptomics datasets.
![image](https://github.com/LiminLi-xjtu/st-Xprop/blob/master/st-Xprop_arch.jpg)

## System Requirements

st-Xprop is implemented in **Python 3.9**.  All experiments reported in the paper were conducted on a workstation equipped with an **NVIDIA GeForce RTX 4090** GPU.

## Dependencies

The following dependencies are required to run st-Xprop:

### Python packages
- `h5py==3.9.0`
- `matplotlib==3.10.8`
- `numpy==1.26.4`
- `pandas==2.2.2`
- `Pillow==12.1.0`
- `POT==0.9.5`
- `rpy2==3.6.2`
- `scanpy==1.12`
- `scikit_learn==1.8.0`
- `scipy==1.11.4`
- `timm==1.0.24`
- `torch==2.4.0`
- `torchvision==0.19.0`
- `tqdm==4.67.1`

### R environment
R >= 4.2

## Example Usage

### Input Data Format

st-Xprop accepts spatial transcriptomics data in .h5ad or .h5 format.
If your data is stored in other formats, please refer to the following resources for conversion: [Scanpy](https://scanpy.readthedocs.io/en/stable/) or [anndata](https://anndata.readthedocs.io/en/stable/)

### Histological Image Preprocessing

Histological image features can be extracted using one of the following methods:
* stMVC-based preprocessing (Preprocessing_stMVC.py)
* Pre-trained Vision Transformer (ViT) (Preprocessing_ViT.py)
  
The choice of method should be specified via the image_type field in the configuration file.

### Data Availability

All datasets used in this study have been curated and deposited on [Zenodo](https://zenodo.org/records/18449464). 


### Configuration

Model hyperparameters and dataset-specific settings are defined in config_stMVC.json.
You may adjust these parameters according to your experimental requirements.

Example parameter settings in the config file:

```bash

lambda_1 – lambda_4 : Weights for different loss terms
lr_pr               : Learning rate for the pretraining stage
lr                  : Learning rate for the main training stage
epoch_pre           : Number of pretraining epochs
epoch               : Number of training epochs
seed                : Random seed for reproducibility
r1                  : Weight of spatial adjacency in clustering module
r2                  : Weight of image adjacency in clustering module
image_type           : Type of image feature (stMVC or ViT)
adj_type             : Spatial adjacency construction method (Radius or KNN)
k_image              : Number of neighbors for image adjacency (KNN)
rad_cutoff           : Radius cutoff for spatial adjacency

```
### Running st-Xprop

To perform spatial domain identification using st-Xprop, run:

```bash
python main.py --name <dataset_name> --slice <slice_name>
```
where:

* `dataset_name` specifies the dataset (e.g., `CHD`)
* `slice_name` specifies the slice identifier (e.g., `D10`, `D14`)

An example workflow is provided in **`example.ipynb`**.

