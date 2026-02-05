import torch
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import timm
import h5py
import numpy as np
import scanpy as sc
from tifffile import imread

def plot_spots_on_image(img, spatial_coords, data_name, img_type='hires', spot_size=5, color='red', alpha=0.6,
                        flip_y=False, title="Spots overlaid on image"):
    """
    Visualizes histological image and spatial coordinates.

    Parameters:
    - img: A 2D NumPy array of images
    - spatial_coords: An (n, 2) NumPy array containing spatial coordinates (x, y)
    - flip_y: If True, the y-coordinate will be flipped
    """
    x = spatial_coords[:, 0]
    y = spatial_coords[:, 1]
    if flip_y:
        y = img.shape[0] - y

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, s=spot_size, c=color, alpha=alpha)
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.savefig("spots_on_image_"+data_name+"_" + img_type+".png", dpi=300, bbox_inches='tight')

def extract_patch_vit_embeddings(
    image_path,
    positions_path,
    scalefactors_path,
    model_path,
    swap=False,
    patch_size=224,
    embedding_type='cls', 
    device='cuda'
):
    assert embedding_type in ['cls', 'mean', 'concat'], "embedding_type must be one of 'cls', 'mean', 'concat'"

    # load hires image
    image = Image.open(image_path).convert('RGB')
    W, H = image.size

    # load spot coordinates
    if swap:
        df = pd.read_csv(positions_path)
        df.columns = ['barcode', 'in_tissue', 'col', 'row', 'pxl_y', 'pxl_x']
        df['in_tissue'] = df['in_tissue'].astype(int)
        df = df[df['in_tissue'] == 1].reset_index(drop=True)
    else:
        df = pd.read_csv(positions_path, header=None)
        df.columns = ['barcode', 'in_tissue', 'row', 'col', 'pxl_x', 'pxl_y']
        df['in_tissue'] = df['in_tissue'].astype(int)
        df = df[df['in_tissue'] == 1].reset_index(drop=True)

    df = df.rename(columns={'pxl_x': 'x', 'pxl_y': 'y'})

    # hires coordinates
    sf = json.load(open(scalefactors_path, 'r'))
    scale = sf['tissue_hires_scalef']
    df['xh'] = df['x'] * scale
    df['yh'] = df['y'] * scale

    # load ViT model
    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        init_values=1e-5,
        dynamic_img_size=True,
        num_classes=0
    )
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    embeddings = []
    for _, row in df.iterrows():
        x, y = int(row['xh']), int(row['yh'])
        half = patch_size // 2
        patch = image.crop((
            max(0, x - half), max(0, y - half),
            min(W, x + half), min(H, y + half)
        ))
        patch_tensor = transform(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            tokens = model.forward_features(patch_tensor)  

        # embedding types
        if embedding_type == 'cls':
            emb = tokens[:, 0, :]  
        elif embedding_type == 'mean':
            emb = tokens[:, 1:, :].mean(dim=1)  
        elif embedding_type == 'concat':
            cls = tokens[:, 0, :]
            mean = tokens[:, 1:, :].mean(dim=1)
            emb = torch.cat([cls, mean], dim=1)  

        embeddings.append(emb.cpu())

    emb_tensor = torch.cat(embeddings, dim=0)  
    return df['barcode'].tolist(), emb_tensor





def save_embeddings_to_h5(barcodes, features, save_path):

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('features', data=features)  

        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('barcodes', data=np.array(barcodes, dtype=dt))

    print(f"Saved embeddings to {save_path}, shape: {features.shape}")


##############################################################################################################


# DLPFC
data_name = 'DLPFC'
slice = '151672'

# # MB
# data_name = 'MB'
# slice = 'MBC'

# # BRCA
# data_name = 'BCRA'
# slice = 'BCRA'

# # CHD
# data_name = 'CHD'
# slice = 'D10'

if data_name=='DLPFC' or data_name=='MB' or data_name=='CHD':
    path = f'../../datasets/{data_name}/{slice}/'
    adata = sc.read_visium(path=path, count_file=slice+'_filtered_feature_bc_matrix.h5')
    coords = adata.obsm["spatial"].astype(int)
elif data_name=='CHD':
    path = f'../../datasets/{data_name}/{slice}/'
    adata = sc.read_h5ad(f'{path}{slice}_reset.h5ad')
    coords = adata.obsm["spatial"].astype(int)
elif data_name=='BCRA':
    path = f'../../datasets/{data_name}/'
    adata = sc.read_visium(path=path, count_file=slice+'_filtered_feature_bc_matrix.h5')
    coords = adata.obsm["spatial"].astype(int)

img = Image.open(f'{path}spatial/tissue_hires_image.png').convert('RGB')
df = pd.read_csv(path + 'spatial/tissue_positions_list.csv', header=None)
with open(path + 'spatial/scalefactors_json.json') as f:
    sf = json.load(f)
scale = sf['tissue_hires_scalef']
coords_scale = coords * scale

plot_spots_on_image(img, coords_scale, slice,img_type='hires')


embedding_types = ['cls', 'mean', 'concat']
for embedding_type in embedding_types:
    barcodes, emb = extract_patch_vit_embeddings(
        image_path=path + 'spatial/tissue_hires_image.png',
        positions_path=path + 'spatial/tissue_positions_list.csv',
        scalefactors_path=path + 'spatial/scalefactors_json.json',
        model_path="../pytorch_model.bin",
        patch_size=224,
        embedding_type=embedding_type,
        device='cuda:0'
    )

    save_embeddings_to_h5(barcodes, emb, path+'vit_patch_' + embedding_type + '_embeddings.h5')


