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


def evaluate_BHS_Standard(ys, preds):
    def _BHS_metric(err):
        leq5 = (err < 5).sum()
        leq10 = (err < 10).sum()
        leq15 = (err < 15).sum()

        return (leq5*100.0/len(err), leq10*100.0/len(err), leq15*100.0/len(err))

    def _calcError(ys, preds):
        sbps = abs(ys.max(axis=1) - preds.max(axis=1)).reshape(-1)
        maps = abs(ys.mean(axis=1) - preds.mean(axis=1)).reshape(-1)
        dbps = abs(ys.min(axis=1) - preds.min(axis=1)).reshape(-1)
        bps = abs(ys - preds).reshape(-1)

        return (sbps, dbps, maps, bps)


    (sbps, dbps, maps, bps) = _calcError(ys, preds)   # compute errors

    sbp_percent = _BHS_metric(sbps)
    dbp_percent = _BHS_metric(dbps)
    map_percent = _BHS_metric(maps)
    bp_percent = _BHS_metric(bps)

    print('----------------------------')
    print('|        BHS-Metric        |')
    print('----------------------------')

    print('----------------------------------------')
    print('|     | <= 5mmHg | <=10mmHg | <=15mmHg |')
    print('----------------------------------------')
    print('| SBP |  {} %  |  {} %  |  {} %  |'.format(round(sbp_percent[0], 2), round(sbp_percent[1], 2), round(sbp_percent[2], 2)))
    print('| MAP |  {} %  |  {} %  |  {} %  |'.format(round(map_percent[0], 2), round(map_percent[1], 2), round(map_percent[2], 2)))
    print('| DBP |  {} %  |  {} %  |  {} %  |'.format(round(dbp_percent[0], 2), round(dbp_percent[1], 2), round(dbp_percent[2], 2)))
    print('| ALL |  {} %  |  {} %  |  {} %  |'.format(round(bp_percent[0], 2), round(bp_percent[1], 2), round(bp_percent[2], 2)))
    print('----------------------------------------')


def evaluate_AAMI_Standard(ys, preds):
    def _calcErrorAAMI(ys, preds):
        sbps = (ys.max(axis=1) - preds.max(axis=1)).reshape(-1)
        maps = (ys.mean(axis=1) - preds.mean(axis=1)).reshape(-1)
        dbps = (ys.min(axis=1) - preds.min(axis=1)).reshape(-1)
        bps = (ys - preds).reshape(-1)

        return (sbps, dbps, maps, bps)

    (sbps, dbps, maps, bps) = _calcErrorAAMI(ys, preds)

    print('---------------------')
    print('|   AAMI Standard   |')
    print('---------------------')

    print('-----------------------')
    print('|     |  ME   |  STD  |')
    print('-----------------------')
    print('| SBP | {} | {} |'.format(round(np.mean(sbps), 3), round(np.std(sbps), 3)))
    print('| MAP | {} | {} |'.format(round(np.mean(maps), 3), round(np.std(maps), 3)))
    print('| DBP | {} | {} |'.format(round(np.mean(dbps), 3), round(np.std(dbps), 3)))
    print('| ALL | {} | {} |'.format(round(np.mean(bps), 3), round(np.std(bps), 3)))
    print('-----------------------')


def inference(te_loader, model, args):
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
        epis = y_std * epis.mean(dim=-1)

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
    te_dataset = ARTDataset(args.te_path)
    te_loader = DataLoader(te_dataset, batch_size=1024, shuffle=False)
    config = get_model_config(args, postprocess=args.postprocess)

    print('config', config)
    model = Model(**config).to(device)
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        xs, ys, preds, epises = inference(te_loader, model, args)

    (SBP_m, MAP_m, DBP_m, ALL_m), (SBP_ci, MAP_ci, DBP_ci, ALL_ci) = check_overall_performance(ys, preds)
    print('[MAE] SBP:{:.3f}+-{:.3f}, MAP:{:.3f}+-{:.3f}, DBP:{:.3f}+-{:.3f}, All:{:.3f}+-{:.3f}'.format(SBP_m, SBP_ci, 
            MAP_m, MAP_ci, DBP_m, DBP_ci, ALL_m, ALL_ci))

    evaluate_BHS_Standard(ys, preds)
    evaluate_AAMI_Standard(ys, preds)

