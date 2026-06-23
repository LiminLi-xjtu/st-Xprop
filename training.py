import tqdm
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans
import os

from utils import *
from loss import *
from eva import *


def cluster_init(model, X_E, A_P, A_I, args):
    model.eval()
    with torch.no_grad():
        output = model(X_E, A_P, A_I, args)
        U = output['U'].cpu().numpy()
    
    kmodel = KMeans(
        n_clusters=args.num_clusters,
        init='k-means++',
        n_init=50,
        random_state=args.seed
    )
    kmodel.fit(U)
    return kmodel.cluster_centers_

def pretrain(model, adata, A_P, A_I, args):
    """
    预训练：只训练 PCA reconstruction loss
    """
    X_E = torch.FloatTensor(adata.obsm['X_pca']).to(args.device)
    print("PreTraining...")
    


    optimizer_pretrain = Adam(
        list(model.gcn_1.parameters()) +
        list(model.gcn_2.parameters()) +
        list(model.encoder_expr.parameters()) +
        list(model.decoder_expr.parameters()) +
        [model.a, model.b, model.alpha],  
        lr=args.lr_pr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pretrain, mode='min', factor=0.75, threshold=0.05,
        threshold_mode='rel', patience=5, min_lr=1e-5
    )

    save_checkpoint_path = f"save/checkpoint/{args.name}/"
    os.makedirs(save_checkpoint_path, exist_ok=True)
    early_stopping_pretrain = EarlyStopping(
        patience=50, verbose=False, checkpoint_file=f"{save_checkpoint_path}pretrain_{args.slice}_{args.image_emb_type}_seed{args.seed}.pth"
    )

    model.train()

    
    for epoch in tqdm.tqdm(range(args.epoch_pre + 1)):
        output = model(X_E, A_P, A_I, args)
        X_E_hat = output['X_E_hat']

        # PCA reconstruction loss
        L_rec = reconstruction_loss(X_E_hat, X_E)
        loss = args.lambda_1 * L_rec

        optimizer_pretrain.zero_grad()
        loss.backward()
        optimizer_pretrain.step()
        scheduler.step(loss.item())

        early_stopping_pretrain(loss.item(), model)
        if early_stopping_pretrain.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    save_model_path = f"save/models/{args.name}/"
    os.makedirs(save_model_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_model_path}{args.slice}_pretrain_{args.image_emb_type}_seed{args.seed}.pkl")
    print("Pretrain model saved.")



def train(model, device, adata, y, A_P, A_I, args):

    X_E = torch.FloatTensor(adata.obsm['X_pca']).to(args.device)
    model.load_state_dict(torch.load(f"save/models/{args.name}/{args.slice}_pretrain_{args.image_emb_type}_seed{args.seed}.pkl"))
    model.to(device)


    centers = cluster_init(model, X_E, A_P, A_I, args)
    model.cluster_centers.data = torch.tensor(centers, dtype=torch.float32).to(args.device)


    optimizer_main = Adam(list(model.gcn_1.parameters()) + 
                          list(model.gcn_2.parameters()) +
                          list(model.encoder_expr.parameters()) +
                          list(model.decoder_expr.parameters()) +
                          [model.a, model.b, model.alpha], lr=args.lr)
    optimizer_cluster = Adam([model.cluster_centers], lr=args.lr_pr)

    model.train()
    save_checkpoint_path = f"save/checkpoint/{args.name}/"
    os.makedirs(save_checkpoint_path, exist_ok=True)
    early_stopping_train = EarlyStopping(
        patience=50, verbose=False, checkpoint_file=f"{save_checkpoint_path}train_{args.slice}_{args.image_emb_type}_seed{args.seed}.pth"
    )

    for epoch in tqdm.tqdm(range(args.epoch + 1)):
        output = model(X_E, A_P, A_I, args)
        X_E_hat, U, U_P, U_I = output['X_E_hat'], output['U'], output['U_P'], output['U_I']
        Q = model.q_distribute(U, U_P, U_I)

        L_rec    = reconstruction_loss(X_E_hat, X_E)
        L_KL     = distribution_loss(Q, target_distribution(Q[0].data))
        A_C      = args.r1 * A_P + args.r2 * A_I
        L_smooth = neighbor_kl_smoothness(Q[0], A_C, symmetric=True)
        L_cut    = mincut_loss(Q[0], A_C)
        loss_main = args.lambda_1 * L_rec
        loss_cluster = args.lambda_2 * L_KL + args.lambda_3 * L_smooth + args.lambda_4 * L_cut
        loss_total = loss_main + loss_cluster

        optimizer_main.zero_grad()
        optimizer_cluster.zero_grad()
        loss_total.backward()
        optimizer_main.step()
        optimizer_cluster.step()

        # Early stopping
        early_stopping_train(loss_total.item(), model)
        if early_stopping_train.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break


    save_model_path = f"save/models/{args.name}/"
    os.makedirs(save_model_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_model_path}{args.slice}_train_{args.image_emb_type}_seed{args.seed}.pkl")

    adata.obsm['st-Xprop'] = U.data.cpu().numpy()
    if args.emb:
        save_res_path = f'save/res/{args.name}/'
        os.makedirs(save_res_path, exist_ok=True)
        np.save(f'{save_res_path}{args.slice}_{args.image_emb_type}.npy', adata.obsm['st-Xprop'])

    results_df = evaluate(adata, args, args.num_clusters, y=y)
    return results_df