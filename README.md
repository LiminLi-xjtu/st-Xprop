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
* **stMVC-based preprocessing (Preprocessing_stMVC.py)
* **Pre-trained Vision Transformer (ViT) (Preprocessing_ViT.py)
The choice of method should be specified via the image_type field in the configuration file.

### Data Availability

All datasets used in this study have been curated and deposited on [Zenodo](https://zenodo.org/records/18449464). 


### Configuration

Model hyperparameters and dataset-specific settings are defined in config.json.
You may adjust these parameters according to your experimental requirements.

Example parameter settings in the config file:

```bash

lambda_1-lambda_4: The weights for loss regularization
lr_pr: The pretraining learning rate for optimization
lr: The training learning rate for optimization
epoch: The number of training epochs
seed: The random seed for reproducibility
r1: The weight of spatial part for adjacency matrix construction in the clustering module
r2: The weight of image part for adjacency matrix construction in the clustering module
image_type: The image feature type that depends on the histoligical image is preprocessed by stMVC or ViT
adj_type: The spatial adjacency matrix is constructed using a radius-based neighbor strategy or k-nearest neighbor (KNN) algorithm
k_image: The number of nearest neighbors used for image adjacency matrix construction
rad_cutoff: The radius used for spatial adjacency matrix construction

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

