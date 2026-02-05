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
    """
    Vectorized KL smoothness loss between neighboring soft cluster assignments.

    q_probs: [N, C] — soft assignment matrix
    A: [N, N] — adjacency matrix
    """
    # Normalize q to be probabilities
    q = q_probs / (q_probs.sum(dim=1, keepdim=True) + eps)
    log_q = torch.log(q + eps)

    # Compute all pairwise KL(i || j)
    # q_i [N, 1, C], q_j [1, N, C] => KL matrix [N, N]
    q_i = q.unsqueeze(1)      # [N, 1, C]
    q_j = q.unsqueeze(0)      # [1, N, C]
    log_q_i = log_q.unsqueeze(1)
    log_q_j = log_q.unsqueeze(0)

    kl_ij = (q_i * (log_q_i - log_q_j)).sum(-1)  # [N, N]
    if symmetric:
        kl_ji = (q_j * (log_q_j - log_q_i)).sum(-1)
        kl_matrix = kl_ij + kl_ji
    else:
        kl_matrix = kl_ij

    # Apply adjacency mask
    masked_kl = kl_matrix * A
    loss = masked_kl.sum() / (A.sum() + eps)
    return loss

def mincut_loss(q_probs, A):
    """
    Minimum cut loss — penalize adjacent nodes assigned to different clusters
    q_probs: [N, C] soft assignment matrix
    A: [N, N] adjacency matrix (binary or weighted)
    """
    q_norm = F.normalize(q_probs, dim=1)  # cosine-like similarity
    sim_matrix = torch.matmul(q_norm, q_norm.T)  # [N, N]
    cut_loss = (A * (1 - sim_matrix)).sum() / (A.sum() + 1e-8)
    return cut_loss

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()
    return mask


def contrastive_loss(batch_size, temperature, z_i, z_j):
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)

    mask = mask_correlated_samples(batch_size)

    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N

    return loss


def spatial_contrastive_loss(Z, A_spatial, temperature=0.1):
    Z = F.normalize(Z, dim=1)
    sim_matrix = torch.matmul(Z, Z.T) / temperature

    mask_pos = A_spatial > 0  # spatial neighbor mask
    mask_self = torch.eye(A_spatial.size(0), dtype=torch.bool, device=Z.device)
    mask_neg = ~mask_pos & ~mask_self

    # Numerators: sum over positive neighbors
    exp_sim = torch.exp(sim_matrix)
    numerator = (exp_sim * mask_pos).sum(dim=1)

    # Denominator: all non-self entries
    denominator = (exp_sim * ~mask_self).sum(dim=1) + 1e-8

    loss = -torch.log(numerator / denominator)
    return loss.mean()


def reconstruction_loss(X_hat, X):
    loss = F.mse_loss(X_hat, X)
    return loss

# loss function (reconstruction + KL clustering placeholder)
def total_loss(X_true, X_recon, Q, P, kl_weight=1.0):

    recon_loss_expr = reconstruction_loss(X_recon, X_true)
    kl_loss = distribution_loss(Q, P)

    return recon_loss_expr + kl_weight * kl_loss
