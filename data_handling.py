import h5py
import numpy as np
import os
from tqdm import tqdm
import pickle
import argparse


def prepare_data(data_path, out_dir):
    val_start = 90000
    val_end = 100000

    fl = h5py.File(data_path, 'r')  

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []

    for i in tqdm(range(0, val_start), desc='Training Data'):
        X_train.append(np.array(fl['data'][i][1]).reshape(1, -1))
        Y_train.append(np.array(fl['data'][i][0]).reshape(1, -1))

    for i in tqdm(range(val_start, val_end), desc='Validation Data'):
        X_val.append(np.array(fl['data'][i][1]).reshape(1, -1))
        Y_val.append(np.array(fl['data'][i][0]).reshape(1, -1))

    for i in tqdm(range(val_end, len(fl['data'])), desc='Test Data'):
        X_test.append(np.array(fl['data'][i][1]).reshape(1, -1))
        Y_test.append(np.array(fl['data'][i][0]).reshape(1, -1))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    X_mean = np.mean(X_train)
    X_std = np.std(X_train, ddof=1)
    Y_mean = np.mean(Y_train)
    Y_std = np.std(Y_train, ddof=1)

    X_train -= X_mean
    X_train /= X_std
    Y_train -= Y_mean
    Y_train /= Y_std
    pickle.dump({'PPG': X_train, 'ABP': Y_train}, open(os.path.join(out_dir, 'train.p'), 'wb'))

    X_val = np.array(X_val)
    X_val -= X_mean
    X_val /= X_std
    Y_val = np.array(Y_val)
    Y_val -= Y_mean
    Y_val /= Y_std
    pickle.dump({'PPG': X_val, 'ABP': Y_val}, open(os.path.join(out_dir, 'valid.p'), 'wb'))

    X_test = np.array(X_test)
    X_test -= X_mean
    X_test /= X_std
    Y_test = np.array(Y_test)
    Y_test -= Y_mean
    Y_test /= Y_std
    pickle.dump({'PPG': X_test, 'ABP': Y_test}, open(os.path.join(out_dir, 'test.p'), 'wb'))

    pickle.dump({'PPG_mean': X_mean,
                'PPG_std': X_std,
                'ABP_mean': Y_mean,
                'ABP_std': Y_std}, open(os.path.join(out_dir, 'stats.p'), 'wb'))


def main(args):
    data_path = args.data_path
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    prepare_data(data_path, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preparation',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-i', type=str, default='DB/data.hdf5', help='Preprocessed data path')
    parser.add_argument('--out_dir', '-o', type=str, default='datasets', help='Out directory')
    args = parser.parse_args()
    main(args)
