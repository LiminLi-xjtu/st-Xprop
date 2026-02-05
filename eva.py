import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import matplotlib
# matplotlib.use("Agg")

from utils import *

def evaluate(adata, args, num_clusters, y=None):
    
    clu_methods = ['kmeans', 'mclust'] #, 'louvain', 'leiden'
    refinement = args.refinement

    if refinement:
        columns = ['kmeans', 'ref_kmeans', 'mclust', 'ref_mclust', 'louvain', 'ref_louvain', 'leiden', 'ref_leiden']
    else:
        columns = clu_methods

    results_df = pd.DataFrame(columns=columns, dtype=float)


    print(f"Evaluating n_clusters = {num_clusters}")
    
    adata.obs[columns]=None

    for clu_method in clu_methods:
        try:
            clustering(adata, n_clusters=num_clusters, used_obsm='st-Xprop', method=clu_method,
                start=args.start if hasattr(args, 'start') and args.start is not None else 0.05,
                end=args.end if hasattr(args, 'end') and args.end is not None else 4,
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
            results_df.loc[num_clusters, clu_method] = None
            if refinement:
                results_df.loc[num_clusters, 'ref_' + clu_method] = None   

    return results_df


def plot(adata, results_df, args, key='ARI'):
    
    columns = results_df.columns
                              
    ncols = 2
    nrows = max(1, (len(columns) + ncols - 1) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))

    # if nrows == 1:
    #     axs = [axs]  
    # else:
    #     axs = axs.flatten()  
        
    if nrows == 1 and ncols == 1:
        axs = np.array([axs])  # 单个图的情况
    elif nrows == 1 or ncols == 1:
        axs = np.array(axs).flatten()
    else:
        axs = axs.flatten()
        
    if args.name=="CHD":
        for i, col in enumerate(columns):
            if results_df[col].values is not None:
                ax = axs[i]
                sc.pl.spatial(adata, color=col, spot_size=180, size=1.3, title='', ax=ax, show=False)
                ax.set_title(f"{col}, {key}={results_df[col].values.round(3)}", fontsize=12)
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_bbox_to_anchor((1, 1))
    else:
        for i, col in enumerate(columns):
            if not results_df[col].isna().all():
                ax = axs[i]
                sc.pl.spatial(adata, color=col, size=1.3, title='', ax=ax, show=False)
                ax.set_title(f"{col}, {key}={results_df[col].values.round(3)}", fontsize=12)
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_bbox_to_anchor((1, 1))
                    
                    

    for j in range(len(columns), len(axs)):
        axs[j].axis('off')

    fig.suptitle(args.slice,fontsize=16)

    fig_path = f"save/figures/{args.name}/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    plt.tight_layout()
    output_path = f"{args.slice}_{args.vit_type}.png"
    fig.savefig(fig_path+output_path, dpi=300, bbox_inches='tight')
