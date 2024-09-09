import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class InstanceLoss(nn.Module):
    def __init__(self, batch_size=256, temperature=1):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [N,N-1]
        loss = self.criterion(logits, labels)  # [N]
        loss /= N

        return loss    

        
class Instance_Align_Loss(nn.Module):
    def __init__(self):
        super(Instance_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss2 = mse(gt, P)

        return Loss2

    
class InsNegLoss(nn.Module):
    def __init__(self, temperature=1):
        super(InsNegLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        
        N_i = z_i.size(0)
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        positive_similarities, positive_indices = torch.max(sim_matrix, dim=1)
        mask = sim_matrix < positive_similarities.unsqueeze(1)
        negative_similarities = torch.zeros_like(sim_matrix)
        negative_similarities[mask] = sim_matrix[mask]

        num_neg_samples_per_row = mask.sum(dim=1, keepdim=True)
        max_neg_samples = num_neg_samples_per_row.max().item()

        negative_similarities_padded = torch.zeros((N_i, max_neg_samples), device=sim_matrix.device)
        for i in range(N_i):
            neg_samples = negative_similarities[i, mask[i]]
            negative_similarities_padded[i, :len(neg_samples)] = neg_samples
        
        exp_pos_sim = torch.exp(positive_similarities)
        
        neg_sim_sum = negative_similarities_padded.sum(dim=1)
        neg_sim_sum = torch.clamp(neg_sim_sum, max=30)

        exp_neg_sim_sum = torch.exp(neg_sim_sum)
        if torch.isinf(exp_neg_sim_sum).any() or torch.isnan(exp_neg_sim_sum).any():
            print("Warning: exp_neg_sim_sum contains inf or nan values.")

        positive_loss_info_nce = -torch.log(exp_pos_sim / exp_neg_sim_sum).mean()
        
        # Triplet Margin Loss
        triplet_loss = F.relu(positive_similarities.unsqueeze(1) - negative_similarities_padded.mean(dim=1, keepdim=True) + 1.0).mean()
        
        # 总损失
        loss = positive_loss_info_nce + triplet_loss
        
        return loss