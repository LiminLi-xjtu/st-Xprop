import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import matplotlib
# matplotlib.use("Agg")

from utils import *

def evaluate(adata, args, num_clusters, y=None):
    
    clu_methods = ['kmeans', 'mclust'] 
    refinement = args.refinement

    if refinement:
        columns = ['kmeans', 'ref_kmeans', 'mclust', 'ref_mclust']
    else:
        columns = clu_methods

    results_df = pd.DataFrame(columns=columns, dtype=float)


    print(f"Evaluating n_clusters = {num_clusters}")
    
    adata.obs[columns]=None

    for clu_method in clu_methods:
        try:
            clustering(adata, n_clusters=num_clusters, used_obsm='st-Xprop', method=clu_method,
                increment=0.01, refinement=refinement
            )
            
            if args.name=="DLPFC":
                adata2 = adata[~pd.isnull(adata.obs['ground_truth'])].copy()
            elif args.name=='HER2ST':
                adata2 = adata[adata.obs['ground_truth'] != 'undetermined'].copy()
            else:
                adata2 = adata.copy()

            if not y.isna().all():
                results_df.loc[0, clu_method] = metrics.adjusted_rand_score(
                    adata2.obs[clu_method], adata2.obs['ground_truth']
                )
                if refinement:
                    results_df.loc[0, 'ref_' + clu_method] = metrics.adjusted_rand_score(
                        adata2.obs['ref_' + clu_method], adata2.obs['ground_truth']
                    )
            else:
                results_df.loc[0, clu_method] = metrics.silhouette_score(
                    adata2.obsm['st-Xprop'], adata2.obs[clu_method]
                )
                if refinement:
                    results_df.loc[0, 'ref_' + clu_method] = metrics.silhouette_score(
                        adata2.obsm['st-Xprop'], adata2.obs['ref_' + clu_method]
                    )
                    
        except Exception as e:
            print(f"{clu_method} failed for n_clusters={num_clusters}: {e}")
            results_df.loc[0, clu_method] = None
            if refinement:
                results_df.loc[0, num_clusters, 'ref_' + clu_method] = None   

    return results_df

