import torch
from torch import optim
from torch.utils.data import DataLoader
import os
import numpy as np
from data_manager import ARTDataset
from model import Model
import argparse
from utils import make_dirs, get_model_config, get_model_description
from args import parse_args
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import os
import scipy.stats

N_SEG = 5


def rescale_value(mu, val):
    scale = mu.max(dim=-1)[0] - mu.min(dim=-1)[0]
    val_norm = val / scale
    return val_norm


def mean_confidence_interval(a, confidence=0.95):
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    ci = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, ci


def check_overall_performance(ys, preds):
    sbps = abs(ys.max(axis=1) - preds.max(axis=1)).reshape(-1)
    maps = abs(ys.mean(axis=1) - preds.mean(axis=1)).reshape(-1)
    dbps = abs(ys.min(axis=1) - preds.min(axis=1)).reshape(-1)
    bps = abs(ys - preds).reshape(-1)

    SBP_m, SBP_ci = mean_confidence_interval(sbps)
    MAP_m, MAP_ci = mean_confidence_interval(maps)
    DBP_m, DBP_ci = mean_confidence_interval(dbps)
    ALL_m, ALL_ci = mean_confidence_interval(bps)
    return (SBP_m, MAP_m, DBP_m, ALL_m), (SBP_ci, MAP_ci, DBP_ci, ALL_ci)


def inference(te_loader, model, save_dir, args):
    print('inference starts!')
    xs = []
    ys = []
    preds =[]
    epises = []

    stats = pickle.load(open(args.stats_path, 'rb'))
    x_mean = stats['PPG_mean']
    x_std = stats['PPG_std']
    y_mean = stats['ABP_mean']
    y_std = stats['ABP_std']

    print('Number of batches:', len(te_loader))
    for itr, (x, y) in enumerate(te_loader):
        print('Iteration: [{} / {}]'.format(itr+1, len(te_loader)))
        B, C, T = y.shape
        x, y = x.to(device), y.to(device)

        pred, epis = model.compute_prediction_and_uncertainty(x)

        x = x.contiguous().view(B * N_SEG, 1, T // N_SEG)
        y = y.contiguous().view(B * N_SEG, 1, T // N_SEG)
        pred = pred.contiguous().view(B * N_SEG, 1, T // N_SEG)
        epis = epis.contiguous().view(B * N_SEG, 1, T // N_SEG)

        x = x_std * x + x_mean
        y = y_std * y + y_mean
        pred = y_std * pred + y_mean
        epis = y_std * epis.mean(dim=-1) # epis: [B*N_SEG, 1]
        epis = rescale_value(pred, epis)

        x = x.permute(0, 2, 1).cpu().numpy()
        y = y.permute(0, 2, 1).cpu().numpy()
        pred = pred.permute(0, 2, 1).cpu().numpy()
        epis = epis.cpu().numpy()

        xs.append(x)
        ys.append(y)
        preds.append(pred)
        epises.append(epis)

    xs = np.vstack(xs)
    ys = np.vstack(ys)
    preds = np.vstack(preds)
    epises = np.vstack(epises).squeeze()

    return xs, ys, preds, epises


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_desc = get_model_description(args, postprocess=args.postprocess)
    checkpoint_path = "params/{}/{}_model.pth".format(model_desc, args.load_type)
    save_dir = "BP_classification/{}/".format(model_desc)
    te_dataset = ARTDataset(args.te_path)
    te_loader = DataLoader(te_dataset, batch_size=1024, shuffle=False)
    config = get_model_config(args, postprocess=args.postprocess)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('config', config)
    model = Model(**config).to(device)
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        xs, ys, preds, epises = inference(te_loader, model, save_dir, args)

    (SBP_m, MAP_m, DBP_m, ALL_m), (SBP_ci, MAP_ci, DBP_ci, ALL_ci) = check_overall_performance(ys, preds)
    print('All - [MAE] SBP:{:.3f}+-{:.3f}, MAP:{:.3f}+-{:.3f}, DBP:{:.3f}+-{:.3f}, All:{:.3f}+-{:.3f}'.format(SBP_m, SBP_ci, 
            MAP_m, MAP_ci, DBP_m, DBP_ci, ALL_m, ALL_ci))
    
    indices = np.argsort(epises)
    xs_sub = []
    ys_sub = []
    preds_sub = []
    sub_len = int(len(xs) * args.subset_ratio)
    for i, idx in enumerate(indices):
        xs_sub.append(xs[idx])
        ys_sub.append(ys[idx])
        preds_sub.append(preds[idx])
        if i+1 == sub_len:
            break
    xs_sub = np.array(xs_sub)
    ys_sub = np.array(ys_sub)
    preds_sub = np.array(preds_sub)
    (SBP_m, MAP_m, DBP_m, ALL_m), (SBP_ci, MAP_ci, DBP_ci, ALL_ci) = check_overall_performance(ys_sub, preds_sub)
    print('Subset - [MAE] SBP:{:.3f}+-{:.3f}, MAP:{:.3f}+-{:.3f}, DBP:{:.3f}+-{:.3f}, All:{:.3f}+-{:.3f}'.format(SBP_m, SBP_ci, 
            MAP_m, MAP_ci, DBP_m, DBP_ci, ALL_m, ALL_ci))
