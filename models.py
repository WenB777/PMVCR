import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from sklearn.utils import shuffle

from instance_loss import InstanceLoss, Instance_Align_Loss, InsNegLoss
from utils import next_batch, shuffle_data, nearest_neighbor_sorting
import evaluation
from scipy.optimize import linear_sum_assignment


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):

        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers = decoder_layers[:-1]
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


def student_t_distribution(z, alpha=1.0):
    """Compute Student's t-distribution for given embeddings."""
    pairwise_distances_squared = torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2)
    q_ij = (1 + pairwise_distances_squared / alpha) ** -((alpha + 1) / 2)
    q_ij /= torch.sum(q_ij, dim=1, keepdim=True)  # Normalize to get probabilities
    return q_ij


class PMVCR_2view(nn.Module):
    def __init__(self, config, device):
        super(PMVCR_2view, self).__init__()
        self._config = config
        self.view = config['training']['view']

        if self._config['Autoencoder']['arch0'][-1] != self._config['Autoencoder']['arch1'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        self.class_dim = config['training']['n_class']

        self.autoencoder0 = Autoencoder(config['Autoencoder']['arch0'], config['Autoencoder']['activations0'],
                                        config['Autoencoder']['batchnorm']).to(device)
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm']).to(device)
        self.ins1 = InstanceLoss(config['training']['batch_size'], temperature=config['training']['temper'])
        self.ins = Instance_Align_Loss().to(device)
        self.ins_N = InsNegLoss(temperature=1)


    def train_pre(self, config, x0_train, x1_train, Y_list, Alignindex_train, Unalignedindex_train, optimizer, epoch, device, lambda_1=10, lambda_2=10, lambda_3=0.01, lambda_4=0.1, wand=False):
        self.autoencoder0.train(), self.autoencoder1.train()

        loss_all = 0
        rec_loss_all = 0
        kl_loss_all = 0
        neg_loss_all = 0

        Y_list = torch.tensor(np.array(Y_list)).int().to(device).detach()
        X0, X1, labels, Alignindex, Unalignedindex = shuffle_data(x0_train, x1_train, Y_list, Alignindex_train, Unalignedindex_train)

        for batch_x0, batch_x1, align_indicator in next_batch(X0, X1, Alignindex, Unalignedindex, config['training']['batch_size']):
            z0 = self.autoencoder0.encoder(batch_x0)
            z1 = self.autoencoder1.encoder(batch_x1)
            rec_loss = F.mse_loss(self.autoencoder0.decoder(z0), batch_x0) + F.mse_loss(self.autoencoder1.decoder(z1), batch_x1)

            z0_align = z0[align_indicator]
            z1_align = z1[align_indicator]

            q0 = student_t_distribution(z0_align)
            q1 = student_t_distribution(z1_align)

            # Compute KL divergence between the distributions
            kl_loss = F.kl_div(q0.log(), q1, reduction='batchmean') + F.kl_div(q1.log(), q0, reduction='batchmean')
            ins_loss1 = self.ins1(z0_align, z1_align)

            z0_unalign = z0[~align_indicator]
            z1_unalign = z1[~align_indicator]

            neg_loss = self.ins_N(z0_align, z1_unalign) + self.ins_N(z1_align, z0_unalign) + self.ins_N(z0_align, z0_unalign) + self.ins_N(z1_align, z1_unalign)

            loss = rec_loss*lambda_1 + ins_loss1*lambda_2 + kl_loss*lambda_3 + neg_loss*lambda_4
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            loss_all += loss.item()
            rec_loss_all += rec_loss.item()
            kl_loss_all += kl_loss.item()
            neg_loss_all += neg_loss.item()

        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss_all / len(x1_train)))


    def train_con_fine(self, config, x0_train, x1_train, Y_list, Alignindex_train, Unalignedindex_train, optimizer, epoch, device, lambda_1=10, lambda_2=1, lambda_3=0.1, wand=False):
        self.autoencoder0.train(), self.autoencoder1.train()

        loss_all= 0
        loss_all_rec = 0
        kl_loss_all = 0
        ins_loss_all = 0

        Y_list = torch.tensor(np.array(Y_list)).int().to(device).detach()
        X0, X1, labels, Alignindex, Unalignedindex = shuffle_data(x0_train, x1_train, Y_list, Alignindex_train, Unalignedindex_train)

        for batch_x0, batch_x1, _ in next_batch(X0, X1, Alignindex, Unalignedindex, config['training']['batch_size']):
            z0 = self.autoencoder0.encoder(batch_x0)
            z1 = self.autoencoder1.encoder(batch_x1)
            rec_loss = F.mse_loss(self.autoencoder0.decoder(z0), batch_x0) + F.mse_loss(self.autoencoder1.decoder(z1), batch_x1)

            Dx = F.cosine_similarity(z0, z1, dim=1)
            gt = torch.ones(z0.shape[0]).to(device)
            ins_loss = self.ins(Dx, gt)

            q0 = student_t_distribution(z0)
            q1 = student_t_distribution(z1)

            # Compute KL divergence between the distributions
            kl_loss = F.kl_div(q0.log(), q1, reduction='batchmean') + F.kl_div(q1.log(), q0, reduction='batchmean')

            loss = rec_loss * lambda_1 + ins_loss * lambda_2 + kl_loss * lambda_3

            loss_all += loss.item()
            loss_all_rec += rec_loss.item()
            kl_loss_all += kl_loss.item()
            ins_loss_all += ins_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss_all / len(x1_train)), 'REC Loss:{:.6f}'.format(loss_all_rec / len(x1_train)), 'KL Loss:{:.6f}'.format(kl_loss_all / len(x1_train), \
                'INS Loss:{:.6f}'.format(ins_loss_all / len(x1_train))))

    def valid(self, X0, X1, Y_list, Alignindex, Unalignedindex, device, wand=False):
        with torch.no_grad():
            self.autoencoder0.eval(), self.autoencoder1.eval()

            Y_list = torch.tensor(np.array(Y_list[1])).int().to(device).detach()

            z0 = self.autoencoder0.encoder(X0)
            z1 = self.autoencoder1.encoder(X1)

            z_both = torch.cat((z0, z1), dim=1).cpu().detach().numpy()

            scores, _ = evaluation.clustering([z_both], Y_list.cpu().detach().numpy())

        print('K-means: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(scores['kmeans']['ACC'], scores['kmeans']['NMI'], scores['kmeans']['ARI']))

        return [scores['kmeans']['ACC'], scores['kmeans']['NMI'], scores['kmeans']['ARI']]
