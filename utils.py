import logging
import torch
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors


def shuffle_data(X0, X1, Y1, Alignindex, Unalignedindex):
    indices = np.arange(len(X0))
    np.random.shuffle(indices)
    
    X0_shuffled = X0[indices]
    X1_shuffled = X1[indices]
    Y1_shuffled = Y1[0][indices]
    
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    Alignindex_shuffled = np.array([idx_map[idx] for idx in Alignindex])
    Unalignedindex_shuffled = np.array([idx_map[idx] for idx in Unalignedindex])
    
    return X0_shuffled, X1_shuffled, Y1_shuffled, Alignindex_shuffled, Unalignedindex_shuffled


def next_batch(X1, X2, Alignindex, Unalignedindex, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        if end_idx - start_idx < 2:
            continue
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]

        idx_tensor = torch.arange(start_idx, end_idx)
        align_tensor = torch.tensor(Alignindex, dtype=torch.long)
        alignment_indicator = torch.zeros(batch_x1.shape[0], dtype=torch.bool)
        alignment_indicator[torch.isin(idx_tensor, align_tensor)] = True


        yield (batch_x1, batch_x2, alignment_indicator)



def nearest_neighbor_sorting(z_unalign, z_align):
    z_unalign_np = z_unalign.cpu().detach().numpy()
    z_align_np = z_align.cpu().detach().numpy()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(z_align_np)
    _, nearest_indices = nbrs.kneighbors(z_unalign_np)

    return nearest_indices.flatten()

def euclidean_dist_2v(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()

    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability  
    return dist