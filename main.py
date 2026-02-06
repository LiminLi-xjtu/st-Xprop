import torch
import numpy as np
import random
import scanpy as sc
from sklearn.decomposition import PCA
import os
import pandas as pd
import sys

from opt_stMVC import get_args
from utils import load_data
from model import Model, Trainer


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    adata, A_P, A_I  = load_data(args)
    X = torch.FloatTensor(adata.obsm["X_pca"])

    model = Model(A_P, A_I, args, n_nodes=X.shape[0], input_dim=X.shape[1]).to(args.device)
    trainer = Trainer(model=model, device=args.device)
    ari_df = trainer.fit(adata, adata.obs["ground_truth"], A_P, A_I, args)

    result_df = ari_df.copy()
    result_df["name"] = args.name
    result_df["slice"] = args.slice 
    result_df["vit_type"] = args.vit_type
    result_df["adj_type"] = args.adj_type
    result_df["lr_pr"] = args.lr_pr
    result_df["lr"] = args.lr
    result_df["rad_cutoff"] = args.rad_cutoff
    result_df["k_spatial"] = args.k_spatial
    result_df["k_image"] = args.k_image
    result_df["r1"] = args.r1
    result_df["r2"] = args.r2
    result_df["lambda_1"] = args.lambda_1
    result_df["lambda_2"] = args.lambda_2
    result_df["lambda_3"] = args.lambda_3
    result_df["lambda_4"] = args.lambda_4

    cols = ["name","slice","vit_type","adj_type","lr_pr","lr","rad_cutoff","k_spatial","k_image","r1","r2",
            "lambda_1","lambda_2","lambda_3","lambda_4"] + [col for col in ari_df.columns]
    result_df = result_df[cols]

    logdir = f"save/eva/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    metrics_log_file = logdir + "metrics_stMVC.csv"

    if not os.path.exists(metrics_log_file):
        result_df.to_csv(metrics_log_file, index=False)
    else:
        result_df.to_csv(metrics_log_file, index=False, mode="a", header=False)


if __name__ == "__main__":

    args = get_args()
    main(args)
