# -*- coding: utf-8 -*-
"""
from stMVC
"""
import stlearn as st
import scanpy as sc
import numpy as np
import pandas as pd
import time
import os
import torch
import random
from pathlib import Path
from stMVC.utilities import parameter_setting
from stMVC.image_processing import tiling
from stMVC.image_SSL import train_simCLR_sImage, extract_representation_simCLR_model
import sys

def Preprocessing( args ):
	start = time.time()
	args.inputPath = Path( args.basePath )

	args.tillingPath = Path( args.basePath + 'tmp/' )
	args.tillingPath.mkdir(parents=True, exist_ok=True)
	args.outPath = Path( args.basePath + 'stMVC/' )
	args.outPath.mkdir(parents=True, exist_ok=True)
	##load spatial transcriptomics and histological data


	adata = sc.read_visium(path=args.inputPath, count_file=args.slice + '_filtered_feature_bc_matrix.h5')
	# adata = sc.read_h5ad(args.basePath + args.slice + '.h5ad')
	adata.var_names_make_unique()
	sc.pp.filter_genes(adata, min_cells=50)
	sc.pp.normalize_total(adata, inplace=True)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

	args.use_cuda = args.use_cuda and torch.cuda.is_available()

	## extract latent features of RNA-seq data by autoencoder-based framework
	print('Start training autoencoder-based framework for learning latent features')
	#RNA_encoding_train(args, adata2, args.basePath + "stMVC/")
	adata  = st.convert_scanpy(adata)


	#save physical location of spots into Spot_location.csv file
	data = { 'imagerow': adata.obs['imagerow'].values.tolist(), 'imagecol': adata.obs['imagecol'].values.tolist() }# 回复
	df   = pd.DataFrame(data, index = adata.obs_names.tolist())
	df.to_csv( args.basePath + 'spatial/' + 'Spot_location.csv' ,mode='w' )

	##tilling histologicald data and train sinCLR model
	print('Tilling spot image')
	tiling(adata, args.tillingPath, target_size = args.sizeImage)
	print('Start training SimCLR model')
	train_simCLR_sImage(args, args.basePath + 'stMVC/' )

	outDir = args.basePath + 'stMVC/'
	save_name_pre = '{}_{}_{}_{}_{}'.format(args.latent_I, args.temperature, args.k, args.batch_size_I, args.max_epoch_I)
	file_model_save = outDir + '/{}_model.pth'.format(save_name_pre)
	extract_representation_simCLR_model(args, outDir = outDir,
										model_file   = file_model_save )

	## extract visual features by ResNet-50 frameowork
	#Extract_representation(args, outDir = args.basePath +  'stMVC/')
	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )


parser  =  parameter_setting()
args    =  parser.parse_args()

if args.name=='DLPFC':
	args.basePath = f'../../datasets/DLPFC/{args.slice}/'
elif args.name=='BRCA':
    args.basePath = f'../../datasets/BRCA/'
elif args.name=='MB':
    args.basePath = f'../../datasets/MB/{args.slice}/'
if args.name=='CHD':
	args.basePath = f'../../datasets/CHD/{args.slice}/'

print(f"Current dataset: {args.slice}")
print(f"Base path: {args.basePath}")

if __name__ == "__main__":
	Preprocessing(args)
