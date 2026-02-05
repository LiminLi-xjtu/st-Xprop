import tqdm
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans
import os

from utils import *
from loss import *
from eva import *

def pretrain(model, adata, A_P, A_I, args):

    X_E = torch.FloatTensor(adata.obsm['X_pca']).to(args.device)

    print("preTraining…")

    optimizer = Adam(model.parameters(), lr=args.lr_pr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, threshold=0.05,
        threshold_mode='rel', patience=5, min_lr=1e-5
    )

    model.train()
    for epoch in tqdm.tqdm(range(args.epoch_pre + 1)):
        # output
        output = model(X_E, A_P, A_I, args)
        X_E_hat = output['X_E_hat']
        # loss: L_{rec}
        L_rec = reconstruction_loss(X_E, X_E_hat)
        loss = args.lambda_1 * L_rec 


        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)


        if epoch == args.epoch_pre:
            save_model_path = f"save/models/{args.name}/"
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            torch.save(model.state_dict(), f"{save_model_path}{args.slice}_pretrain_{args.vit_type}.pkl")





def cluster_init(model, X_E, A_P, A_I, args):

    # calculate embedding similarity
    model.eval()
    with torch.no_grad():
        output = model(X_E, A_P, A_P, args)
        U = output['U']
    # calculate cluster centers
    kmodel = KMeans(n_clusters=args.num_clusters, n_init=20)
    kmodel.fit_predict(U.data.cpu().numpy())
    return kmodel.cluster_centers_


def train(model, device, adata, y, A_P, A_I, args):

    X_E = torch.FloatTensor(adata.obsm['X_pca']).to(args.device)

    print("Training…")
    save_model_path = f"save/models/{args.name}/"
    model.load_state_dict(torch.load(f"{save_model_path}{args.slice}_pretrain_{args.vit_type}.pkl")) #, map_location='cpu'
    model.to(device)
    # calculate embedding similarity and cluster centers
    centers = cluster_init(model, X_E, A_P, A_I, args)
    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-2) # , weight_decay=1e-4

    model.train()
    for epoch in tqdm.tqdm(range(args.epoch+1)):
        
        # output
        output = model(X_E, A_P, A_I, args)
        X_E_hat, U, Q = output['X_E_hat'], output['U'], output['Q']
        # A_hat = output['A_recon']
        
        # loss
        L_rec = reconstruction_loss(X_E_hat, X_E)
        # L_REC2 = 0.0*F.binary_cross_entropy_with_logits(A_hat, A_P)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        A_C = args.r1*A_P + args.r2*A_I
        
        L_smooth = neighbor_kl_smoothness(Q[0], A_C, symmetric=True)
        L_cut = mincut_loss(Q[0], A_C)
        loss =  args.lambda_1 * L_rec +args.lambda_2 * L_KL + args.lambda_3 * L_smooth + args.lambda_4 * L_cut

        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch == args.epoch:
            print("Running epochs: ", epoch)


            save_model_path = f"save/models/{args.name}/"
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            torch.save(model.state_dict(),  f"{save_model_path}{args.slice}_train_{args.vit_type}.pkl")

            adata.obsm['st-Xprop'] = U.data.cpu().numpy()


            if args.emb:
                save_res_path = f'save/res/{args.name}/'
                if not os.path.exists(save_res_path):
                    os.makedirs(save_res_path)
                np.save(f'{save_res_path}{args.slice}_{args.vit_type}.npy', adata.obsm['st-Xprop'])
                

            results_df = evaluate(adata, args, args.num_clusters, y=y)        
            # plot(adata, results_df, args)      

            return results_df






