import torch
from torch import optim
from torch.utils.data import DataLoader
import os
import numpy as np
from math import pi, log
from data_manager import ARTDataset
from model import Model
import json
import time
import argparse
from utils import make_dirs, get_model_config, get_model_description
from args import parse_args
from random import sample
import pickle
import matplotlib.pyplot as plt

n_stds = 4
viz_start_idx = 300
sr = 125

def compute_epistemic_uncertainty(alpha, beta, v):
    epis = (beta / (v * (alpha - 1))) ** 0.5
    epis = torch.clamp(epis, max=1e3)
    print(epis.max())
    return epis


def compute_aleatoric_uncertainty(alpha, beta):
    alea = (beta / ((alpha - 1))) ** 0.5
    alea = torch.clamp(alea, max=1e3)
    return alea

    
def rescale_value(mu, val):
    val_norm = val / mu.var()
    return val_norm


def prepare_to_visualize(y, pred, epis, mean, std):
    y = y.squeeze().cpu().numpy() * std + mean
    pred = pred.squeeze().cpu().numpy() * std + mean
    epis = epis.squeeze().cpu().numpy() * std
    return y, pred, epis


def add_subplot(fig, pos, y, mu, var):
    ax = fig.add_subplot(pos)
    T = len(y)
    t = np.arange(T) / sr
    ax.plot(t, y, c='#EE0000', linestyle='--', zorder=2, label="Ground-truth")
    ax.plot(t, mu, c='k', zorder=3, label="Prediction")
    for k in np.linspace(0, n_stds, 4):
        ax.fill_between(
            t, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#94b5c0',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    ax.set_ylim(20, 220)
    plt.xlabel('Time (seconds)', fontsize=10)
    plt.ylabel('Blood pressure (mmHg)', fontsize=10)
    plt.tight_layout()
    ax.legend(loc="upper right", fontsize='x-small')


def viz_regression(te_loader, model, args):
    dt = pickle.load(open(args.stats_path, 'rb'))
    mean = dt['ABP_mean']
    std = dt['ABP_std']

    for itr, (x, y) in enumerate(te_loader):
        print('itr', itr, flush=True)
        x, y = x.to(device), y.to(device)
        pred, epis = model.compute_prediction_and_uncertainty(x)
        y, pred, epis = prepare_to_visualize(y, pred, epis, mean, std)
        fig = plt.figure(figsize=(5, 2.5), dpi=300)
        plt.subplots_adjust(wspace=0.25, hspace=0.4)
        add_subplot(fig, 111, y, pred, epis)
        plt.show()
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    postprocess = args.postprocess
    model_desc = get_model_description(args, postprocess)
    checkpoint_path = "params/{}/{}_model.pth".format(model_desc, args.load_type)
    te_dataset = ARTDataset(args.te_path)
    te_loader = DataLoader(te_dataset, batch_size=1, shuffle=False, num_workers=1)
    config = get_model_config(args, postprocess)
    model = Model(**config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Load checkpoint from: {}".format(checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        viz_regression(te_loader, model, args)
