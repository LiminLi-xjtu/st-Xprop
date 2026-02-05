import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import Parameter

from training import train, pretrain
from utils import EarlyStopping
from gclayer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, input_dim, out_dim, init, gcn_layers, activation = nn.ELU(), residual=False, k_hop=1): #nn.ELU()
        super(GCN, self).__init__()

        self.activation = activation
        self.input_dim = input_dim
        self.residual = residual
        self.k_hop = k_hop

        self.gcn_layers = []
        self.gcn_layers.append(GraphConvolution(input_dim, out_dim, bias=True))
        for i in range(gcn_layers-1):
            self.gcn_layers.append(GraphConvolution(out_dim, out_dim, bias=True))
        if init:
            self.gcn_layers.append(GraphConvolution(out_dim, out_dim, bias=True))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

    def compute_khop_adj(self, adj, k):
        A_k = adj.clone()
        A_current = adj.clone()
        for _ in range(1, k):
            A_current = torch.matmul(A_current, adj)
            A_k = torch.clamp(A_k + A_current, 0, 1)
        A_k.fill_diagonal_(0)
        return A_k

    def forward(self, X_E, adj):
        if self.k_hop > 1:
            adj = self.compute_khop_adj(adj, self.k_hop)

        out = self.activation(self.gcn_layers[0](self.activation(X_E), adj))
        for gcn in self.gcn_layers[1:]:
            if self.residual:
                out = self.activation(gcn(out, adj)) + out
            else:
                out = self.activation(gcn(out, adj))
        return out


class Model(nn.Module):
    def __init__(self, A_P, A_I, args, n_nodes, input_dim, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.A_P, self.A_I = A_P, A_I

        # GCN encoder
        self.gcn_1 = GCN(input_dim, latent_dim, init=True, gcn_layers=1, k_hop=1, residual=False)
        self.gcn_2 = GCN(latent_dim, latent_dim, init=False, gcn_layers=1, k_hop=1)
        
        self.gcn_3 = GCN(input_dim, latent_dim, init=True, gcn_layers=1)
        self.gcn_4 = GCN(latent_dim, latent_dim, init=False, gcn_layers=1)

        # Gene expression encoder (MLP)
        self.encoder_expr = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder for gene expression reconstruction
        self.decoder_expr = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Decoder for adjacency matrix reconstruction (inner product)
        # self.decoder_adj = lambda Z: torch.sigmoid(Z @ Z.T)

        self.cluster_centers = Parameter(torch.Tensor(args.num_clusters, latent_dim), requires_grad=True)

        # fusion parameter from DFCN
        self.a = Parameter(nn.init.constant_(torch.zeros(n_nodes, latent_dim), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_nodes, latent_dim), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.tensor(0.5), requires_grad=True)

    def q_distribute(self, Z, U_E, Z_G):

        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(U_E.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_gae = 1.0 / (1.0 + torch.sum(torch.pow(Z_G.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_gae = (q_gae.t() / torch.sum(q_gae, 1)).t()

        return [q, q_ae, q_gae]


    def forward(self, X_E, A_P, A_I, args):

        U_P = X_E
        U_I = X_E
        
        
        ###Cross-propagative learning module
        # Spatial view
        U_P = self.gcn_1(U_P, A_P)
        U_P = F.relu(U_P)

        # Image view
        U_I = self.gcn_1(U_I, A_I)
        U_I = F.relu(U_I)

        U_CP = self.gcn_2(U_I, A_P)
        U_I = self.gcn_2(U_P, A_I)
        U_P = U_CP

        # Gene expression encoder
        U_E = self.encoder_expr(X_E)

        # fusion
        U_G = (U_P + U_I) / 2
     
        U_C = self.a * U_E + self.b * U_G
        A_C = args.r1*A_P + args.r2*A_I
        U_l = torch.spmm(A_C, U_C)
        S = torch.mm(U_l, U_l.t())
        S = F.softmax(S, dim=1)
        U_g = torch.mm(S, U_l)
        U = self.alpha * U_g + U_l


        # Decode for gene expression reconstruction
        X_E_hat = self.decoder_expr(U)
        
        # Decode for structure reconstruction
        # A_recon = self.decoder_adj(U)

        Q = self.q_distribute(U, U_E, U_G)

        return {
            'U': U,
            'Q': Q,
            'X_E_hat': X_E_hat
            # 'A_recon': A_recon
        }


class Trainer():

    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)

    def fit(self, adata, y, A_P, A_I, args):

        early_stopping_pretrain = EarlyStopping(patience=20, verbose=False, checkpoint_file='checkpoint_pretrain.pth')
        pretrain(self.model, adata, A_P, A_I, args)
        if y is not None:
            eva_df=train(self.model, self.device, adata, y, A_P, A_I, args)
            return eva_df


