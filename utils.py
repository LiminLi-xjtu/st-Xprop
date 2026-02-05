import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import scanpy as sc
from sklearn.decomposition import PCA
import ot

from adj import adj, normalize_adj
from load_img_vit import load_embeddings_from_h5

def load_data(args):
    dataset_name=args.name
    dataset_slice=args.slice
    
    if dataset_name=='DLPFC' or dataset_name=='MB' or dataset_name=='CHD' or dataset_name=='HER2ST' or dataset_name=='PDAC':
        load_path = f"../../datasets/{dataset_name}/{dataset_slice}/"
    else:
        load_path = f"../../datasets/{dataset_name}/"
        
    # load data
    if dataset_name=='DLPFC':
        adata = sc.read_visium(path=load_path, count_file=f"{args.slice}_filtered_feature_bc_matrix.h5")
        Ann_df = pd.read_csv(f"{load_path}{args.slice}_truth.txt", sep="\t", header=None, index_col=0)
        Ann_df.columns = ["Ground Truth"]
        adata.obs["ground_truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]
    if dataset_name=='MB':
        adata = sc.read_visium(path=load_path, count_file=f"{args.slice}_filtered_feature_bc_matrix.h5")
        adata.obs["ground_truth"] = None
    elif dataset_name=='CHD':
        adata = sc.read_h5ad(f'{load_path}{args.slice}_reset.h5ad')
        adata.obs_names = adata.obs_names + '-1'
        adata.obs["ground_truth"] = adata.obs["region"]
    elif dataset_name=='BRCA':
        adata = sc.read_visium(path=load_path, count_file=f"{args.slice}_filtered_feature_bc_matrix.h5")
        Ann_df = pd.read_csv(f"{load_path}{args.slice}_truth.txt", sep="\t", header=None, index_col=0)
        Ann_df.columns = ["Ground Truth"]
        adata.obs["ground_truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]
    elif dataset_name=='HER2ST':
        adata = sc.read_h5ad(f'{load_path}{args.slice}.h5ad')
        from scipy.sparse import csr_matrix
        adata.X = csr_matrix(adata.X)
        adata.obs["ground_truth"] = adata.obs["annotation"]
    elif dataset_name=='PDAC':
        adata = sc.read_h5ad(f'{load_path}{args.slice}.h5ad')
        adata.obs["ground_truth"] = adata.obs["region"]
        
    get_process(adata,pca_n=50)
    
    if args.vit_type=='stMVC':
        data_image_csv = pd.read_csv(f"{load_path}stMVC.csv", index_col=0) 
        barcode_to_index = {barcode: i for i, barcode in enumerate(data_image_csv.index)}
        common_barcodes = [bc for bc in adata.obs_names if bc in barcode_to_index]
        reordered_indices = [barcode_to_index[bc] for bc in common_barcodes]
        datai_index = data_image_csv.iloc[reordered_indices, :]
        data_image = datai_index.values 
        adata = adata[np.array(common_barcodes), :]
    else:
        h5_path = f"{load_path}{args.vit_type}_embeddings.h5"
        data_image_h5 = load_embeddings_from_h5(h5_path)
        barcode_to_index = {barcode: i for i, barcode in enumerate(data_image_h5[0])}
        common_barcodes = [bc for bc in adata.obs_names if bc in barcode_to_index]
        reordered_indices = [barcode_to_index[bc] for bc in common_barcodes]
        data_image = data_image_h5[1][reordered_indices, :]
        adata = adata[np.array(common_barcodes), :]
        
    # adj for spalial coordinates
    A_P = adj(adata,view="gene",model=args.adj_type,rad_cutoff=args.rad_cutoff,k_cutoff=args.k_spatial)
    A_P = normalize_adj(A_P)
    A_P = torch.FloatTensor(A_P).to(args.device)

    # adj for image
    pca = PCA(n_components=100)
    image_spatial=pca.fit_transform(data_image)
    adata.obsm["image_spatial"]=image_spatial
    A_I = adj(adata,view="image",model="KNN",rad_cutoff=args.rad_cutoff,k_cutoff=args.k_image)
    A_I = normalize_adj(A_I)
    A_I = torch.FloatTensor(A_I).to(args.device)

    if not adata.obs["ground_truth"].isna().all():
        if dataset_name=='DLPFC' or dataset_name=='HER2ST':
            args.num_clusters = len(np.unique(adata.obs["ground_truth"].astype(str)))-1
        else:
            args.num_clusters = len(np.unique(adata.obs["ground_truth"]))
    else:
        args.num_clusters=16
    
    
    return adata, A_P, A_I 
    


def load_embeddings_from_h5(h5_path):
    import h5py
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        barcodes = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['barcodes'][:]]
    return barcodes, features

def get_process(adata,pca_n):
    adata.var_names_make_unique()
    sc.pp.filter_genes_dispersion(adata, n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.tl.pca(adata, n_comps=50)
    X = adata.X.toarray()
    pca_x = PCA(n_components=pca_n)
    X=pca_x.fit_transform(X)
    adata.obsm["X_pca"] = X
   
    return adata
    

# loss

def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P

def reconstruction_loss(X_hat, X):
    loss = F.mse_loss(X_hat, X)
    return loss

def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction="batchmean")
    return loss


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    robjects.r.library("mclust")

    # rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]

    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # mclust_res = np.array(res[-2])

    with localconverter(robjects.default_converter + numpy2ri.converter):
        r_data = robjects.conversion.py2rpy(adata.obsm[used_obsm])
        res = rmclust(r_data, num_cluster, modelNames)
    # mclust_res = np.array(res.rx2("classification"))
    classification_idx = list(res.names()).index("classification")
    mclust_res = np.array(res[classification_idx])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int")
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")
    return adata


def refine_label(adata, radius=50, key="label"):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm["spatial"]
    distance = ot.dist(position, position, metric="euclidean")

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs["label_refined"] = np.array(new_type)

    return new_type


def clustering_kmeans(adata, n_clusters=7, radius=50, used_obsm="st-Xprop", method="kmeans", start=0.1, end=3.0, increment=0.01,
               refinement=True):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(adata.obsm[used_obsm])
    clu = kmeans.labels_.astype(str)
    adata.obs["kmeans"] = clu

    if refinement:
        ref_clu = refine_label(adata, radius, key=method)

    return clu, ref_clu


def clustering(adata, n_clusters=7, radius=50, used_obsm="st-Xprop", method="mclust", start=0.1, end=3.0, increment=0.01,
               refinement=False):

    # pca = PCA(n_components=20, random_state=42)
    # embedding = pca.fit_transform(adata.obsm["emb"].copy())
    # adata.obsm["emb_pca"] = embedding

    if method == "mclust":
        adata = mclust_R(adata, used_obsm=used_obsm, num_cluster=n_clusters)
    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(adata.obsm[used_obsm])
        adata.obs["kmeans"] = kmeans.labels_.astype(str)
    elif method == "leiden":
        res = search_res(adata, n_clusters, use_rep=used_obsm, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
    elif method == "louvain":
        res = search_res(adata, n_clusters, use_rep=used_obsm, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)

    if refinement:
        new_type = refine_label(adata, radius, key=method)
        adata.obs[f"ref_{method}"] = new_type


def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.1, end=3.0, increment=0.01):

    print("Searching resolution...")
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
            print("resolution={}, cluster number={}".format(res, count_unique))
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
            print("resolution={}, cluster number={}".format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            print("Resolution Found: ", method)
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res





class EarlyStopping:

    def __init__(self, patience=10, verbose=False, checkpoint_file=""):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f"Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss


