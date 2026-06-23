import torch
import torch.nn as nn
import torch.nn.functional as F



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


def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss





def neighbor_kl_smoothness(q_probs, A, symmetric=True, eps=1e-10):
    # 1. Basic probability normalization
    q = q_probs / (q_probs.sum(dim=1, keepdim=True) + eps)
    log_q = torch.log(q + eps)
    
    term1 = (q * log_q).sum(dim=1) 

    if A.is_sparse:
        indices = A._indices()
        src, dst = indices[0], indices[1]
        edge_weights = A._values()
        cross_term_ij = (q[src] * log_q[dst]).sum(dim=-1)
        kl_ij_vals = term1[src] - cross_term_ij

        if symmetric:
            cross_term_ji = (q[dst] * log_q[src]).sum(dim=-1)
            kl_ji_vals = term1[dst] - cross_term_ji
            kl_vals = kl_ij_vals + kl_ji_vals
        else:
            kl_vals = kl_ij_vals

        a_sum = A.sum()
        loss = (kl_vals * edge_weights).sum() / (a_sum + eps)

    else:
        term2 = torch.mm(q, log_q.t()) 
        a_sum = A.sum()

        if symmetric:
            row_sums = A.sum(dim=1)  
            sum_term1 = 2.0 * (term1 * row_sums).sum()
            sum_term2 = (term2 * A).sum() + (term2.t() * A).sum()
            loss = (sum_term1 - sum_term2) / (a_sum + eps)
        else:
            row_sums = A.sum(dim=1)  
            sum_term1 = (term1 * row_sums).sum()
            sum_term2 = (term2 * A).sum()
            loss = (sum_term1 - sum_term2) / (a_sum + eps)

    return loss



def mincut_loss(q_probs, A):
    q_norm = F.normalize(q_probs, dim=1)

    if A.is_sparse:
        indices = A._indices()
        src, dst = indices[0], indices[1]
        sim_vals = (q_norm[src] * q_norm[dst]).sum(dim=-1)   
        cut_loss = ((1.0 - sim_vals) * A._values()).sum() / (A.sum() + 1e-8)
    else:
        sim_matrix = torch.mm(q_norm, q_norm.t())        
        cut_loss = (A * (1.0 - sim_matrix)).sum() / (A.sum() + 1e-8)

    return cut_loss




def reconstruction_loss(X_hat, X):
    loss = F.mse_loss(X_hat, X)
    return loss

