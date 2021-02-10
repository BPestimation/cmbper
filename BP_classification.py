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


N_SEG = 5


def classify_BP(ys, preds, title, split='SBP'):
    cls_gt = []
    cls_pred = []
    fig = plt.figure(figsize=(4, 4), dpi=200)
    
    if split == 'SBP':
        boundary1 = 120
        boundary2 = 140
    elif split == 'DBP':
        boundary1 = 80
        boundary2 = 90
    else:
        assert False, 'Split should be SBP or DBP.'

    for i in (range(len(ys))):
        y = ys[i].ravel()
        pred = preds[i].ravel()

        if split == 'SBP':
            gt = max(y)
            pred = max(pred)
        else:
            gt = min(y)
            pred = min(pred)

        if(gt <= boundary1):
            cls_gt.append('Normo.')
        elif((gt > boundary1)and(gt <= boundary2)):
            cls_gt.append('Prehyp.')
        elif(gt > boundary2):
            cls_gt.append('Hyper.')
        else:
            print('bump')

        if(pred <= boundary1):
            cls_pred.append('Normo.')
        elif((pred > boundary1)and(pred <= boundary2)):
            cls_pred.append('Prehyp.')
        elif(pred > boundary2):
            cls_pred.append('Hyper.')
        else:
            print('bump')


    classes = ['Hyper.', 'Prehyp.', 'Normo.']
    print('{} Classification Accuracy'.format(split))
    print(classification_report(cls_gt, cls_pred, labels=classes, digits=5))
    print('Accuracy score:', accuracy_score(cls_gt,cls_pred))

    cm = confusion_matrix(cls_gt, cls_pred, labels=classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = plt.subplot(1,1,1)
    im = ax.imshow(cm, interpolation='nearest', cmap='GnBu')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.25)
    ax.figure.colorbar(im, cax=cax)
    cax.tick_params(labelsize=6)

    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes)


    ax.set_title(title, fontsize=12)
    ax.set_ylabel('True', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=90, fontsize=7, va="center")
    plt.setp(ax.get_xticklabels(), fontsize=7)
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    ax.grid(False)
    fig.tight_layout()
    plt.show()
    plt.close()


def rescale_value(mu, val):
    scale = mu.max(dim=-1)[0] - mu.min(dim=-1)[0]
    val_norm = val / scale
    return val_norm


def compute_epistemic_uncertainty(alpha, beta, v):
    epis = (beta / (v * (alpha - 1))) ** 0.5
    return epis


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
    te_dataset = ARTDataset(args.te_path)
    te_loader = DataLoader(te_dataset, batch_size=1024, shuffle=False)
    config = get_model_config(args, postprocess=args.postprocess)
    model = Model(**config).to(device)
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        xs, ys, preds, epises = inference(te_loader, model, args)

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

    classify_BP(ys, preds, title='All Samples', split='SBP')
    classify_BP(ys_sub, preds_sub, title='Subset', split='SBP')
