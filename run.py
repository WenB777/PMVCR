from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.optim import Adam
import itertools
from queue import Queue
from instance_loss import InstanceLoss
from models import PMVCR_2view
import os
from data_loader import load_data, data_process
import logging
from configure import get_default_config
from sklearn.decomposition import PCA
from utils import euclidean_dist_2v
import random

# Configure logging
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main():
    # Ensure environment variables and other setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_cuda = torch.cuda.is_available()
    logging.info(f"GPU available: {use_cuda}")
    
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    best_scores_kmeans = [0, 0, 0, 0]

    args = parser.parse_args()
    data_name = ['Scene15', 'Reuters_dim10', 'BDGP', 'RGBD']
    logging.info(f"Using dataset: {data_name[args.dataset]}")

    # Set up logging file
    log_path = f"{data_name[args.dataset]}.log"
    setup_logging(log_path)
    
    args.cuda = torch.cuda.is_available()
    logging.info(f"Using CUDA: {args.cuda}")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    config = get_default_config(data_name[args.dataset])
    config['training']['temper'] = args.temper
    manual_seed = config['training']['seed']
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    # Load and process data
    data, label = load_data(data_name[args.dataset])
    dataV0, dataV1, Alignindex, Unalignedindex, label = data_process(data, label, manual_seed, args)

    # Initialize model and optimizer
    PMVCR = PMVCR_2view(config, device)
    optimizer = torch.optim.Adam(
                            itertools.chain(PMVCR.parameters()),
                            lr=args.lr_train,
                        )

    # Pre-training
    epoch = 1
    logging.info('Starting pre-training...')
    while epoch <= args.pre_epochs:
        PMVCR.train_pre(config, dataV0, dataV1, label, Alignindex, Unalignedindex, optimizer, epoch, device, config['training']['lambda11'], config['training']['lambda12'], config['training']['lambda13'], config['training']['lambda14'], wand=False)
        with torch.no_grad():
            scores = PMVCR.valid(dataV0, dataV1, label, Alignindex, Unalignedindex, device, wand=False)
        epoch += 1
    
    logging.info('Starting re-range...')
    dataV0_reranged = dataV0.clone()
    Z0_all, Z1_all = PMVCR.autoencoder0.encoder(dataV0), PMVCR.autoencoder1.encoder(dataV1)

    C = euclidean_dist_2v(Z0_all[Unalignedindex], Z1_all[Unalignedindex])
    for i in range(len(Unalignedindex)):
        idx = torch.argsort(C[:, i])
        dataV0_reranged[Unalignedindex[i]] = dataV0[Unalignedindex][idx[0]]

    logging.info('Starting con-training...')
    while epoch <= args.pre_epochs + args.con_epochs:
        PMVCR.train_con_fine(config, dataV0_reranged, dataV1, label, Alignindex, Unalignedindex, optimizer, epoch, device, config['training']['lambda21'], config['training']['lambda22'], config['training']['lambda23'], wand=False)

        if epoch % 1 == 0:
            with torch.no_grad():
                scores = PMVCR.valid(dataV0_reranged, dataV1, label, Alignindex, Unalignedindex, device, wand=False)
                if scores[0] > best_scores_kmeans[0]:
                    best_scores_kmeans = scores

        epoch += 1
    logging.info('K-means Clustering Results: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(best_scores_kmeans[0], best_scores_kmeans[1], best_scores_kmeans[2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_z', default=32, type=int, help='choose from [32, 64]')
    parser.add_argument('--lr_train', default=0.002, type=float, help='choose from [0.0001~0.001]')
    parser.add_argument('--batch_size', default=512, type=int, help='choose from [512, 1024, 2048]')  # fix
    parser.add_argument('--n_p', default=5, type=int, help='number of positive pairs for each sample')
    # Data
    parser.add_argument('--dataset', default='3', type=int,
                        help='choose dataset from 0-Scene15, 1-Reuters, 2-BDGP, 3-RGBD')
    parser.add_argument('--aligned_p', default='0.5', type=float,
                        help='originally aligned proportions in the partially view-aligned data')
    parser.add_argument('--main_view', default=1, type=int,
                        help='main view to obtain the final clustering assignments, from[0, 1]')
    # Train
    parser.add_argument('--pre_epochs', type=int, default=100)
    parser.add_argument('--con_epochs', type=int, default=100)
    parser.add_argument('--temper', type=float, default=0.5)

    args = parser.parse_args()
    
    main()
