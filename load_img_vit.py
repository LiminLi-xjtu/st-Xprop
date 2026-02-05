

import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
from torchvision.models.feature_extraction import create_feature_extractor
import timm


import torch
import pandas as pd
import json
from PIL import Image
from torchvision import transforms
import timm


import h5py
import numpy as np
def load_embeddings_from_h5(h5_path):

    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        barcodes = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['barcodes'][:]]
    return barcodes, features


def load_tif_embeddings_from_h5(h5_path):

    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
    return features

# 按 patch 裁剪提 embedding
# embedding_types = ['cls', 'mean', 'concat']
# for embedding_type in embedding_types:
#     barcodes1, emb = extract_patch_vit_embeddings(
#         image_path='../../datasets/DLPFC/151675/spatial/tissue_hires_image.png',
#         positions_path='../../datasets/DLPFC/151675/spatial/tissue_positions_list.csv',
#         scalefactors_path='../../datasets/DLPFC/151675/spatial/scalefactors_json.json',
#         patch_size=224,
#         embedding_type=embedding_type,
#         device='cuda'
#     )
#
#     save_embeddings_to_h5(barcodes1, emb, '../../datasets/DLPFC/151675/vit_patch_' + embedding_type + '_embeddings.h5')


# 整图提特征图，再 ROI 聚合
# barcodes2, features2 = extract_roi_vit_embeddings(
#     image_path='../../datasets/DLPFC/151675/spatial/tissue_hires_image.png',
#     positions_path='../../datasets/DLPFC/151675/spatial/tissue_positions_list.csv',
#     scalefactors_path='../../datasets/DLPFC/151675/spatial/scalefactors_json.json',
#     roi_size=7,
#     device='cuda'
# )
#
# save_embeddings_to_h5(barcodes2, features2, '../../datasets/DLPFC/151675/vit_roi_embeddings.h5')
#
# path = '../../datasets/DLPFC/151675/'
# vit_type = 'vit_roi' # 'vit_patch_cls', 'vit_patch_mean', 'vit_patch_concat'
# h5_path = path + vit_type + '_embeddings.h5'
# img_emb = load_embeddings_from_h5(h5_path)
