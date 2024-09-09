import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from idecutils import normalize, aligned_data_split
from sklearn.preprocessing import StandardScaler
import torch
import random
from numpy.random import permutation


def load_data(dataset):
    label = []
    if dataset == 'Scene15':
        mat = sio.loadmat('./datasets/' + dataset + '.mat')
        data = mat['X'][0][0:2]  # 20, 59
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10':
        data = []  # 18758
        mat = sio.loadmat('datasets/Reuters_dim10.mat')
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'BDGP':
        mat = sio.loadmat('datasets/BDGP.mat')
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'][0]).T
    elif dataset == 'RGBD':
        mat = sio.loadmat('datasets/RGB-D.mat')
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'][0]).T
    label = label.reshape(-1)
    label = np.array(label, 'float64')
    y = label
    if dataset in ['Reuters_dim10', 'NoisyMNIST-30000']:
        x = np.array(data)
    else:
        x = np.empty(len(data), dtype=object)
        for i, arr in enumerate(data):
            x[i] = arr

    return x, y


def data_process(data, label, seed, args):
    aligned_sample_index, unaligned_sample_index = aligned_data_split(len(label), args.aligned_p, seed)
    shuffle_index = permutation(unaligned_sample_index)
    auxiliary_view = abs(args.main_view - 1)
    print('auxiliary_view: ', auxiliary_view)  # 打印 辅助视图编号
    data[auxiliary_view][unaligned_sample_index] = data[auxiliary_view][shuffle_index]

    y = np.empty(shape=(2, len(label)))
    y[0] = y[1] = label
    y[auxiliary_view][unaligned_sample_index] = y[auxiliary_view][shuffle_index]

    data0 = np.array(data[0], 'float64')
    data1 = np.array(data[1], 'float64')

    data0 = torch.Tensor(data0).to(args.device)
    data1 = torch.Tensor(data1).to(args.device)
    
    return data0, data1, aligned_sample_index, unaligned_sample_index, y

