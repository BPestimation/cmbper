import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi, log


def Student_NLL(value, df, loc, scale):
    y = (value - loc) / scale
    Z = (scale.log() +
            0.5 * df.log() +
            0.5 * log(pi) +
            torch.lgamma(0.5 * df) -
            torch.lgamma(0.5 * (df + 1.)))
    log_prob = -0.5 * (df + 1.) * torch.log1p(y**2. / df) - Z
    nll = -log_prob.mean()
    return nll


def NIG_Reg(y, gamma, v, alpha, beta):
    error = torch.abs(y-gamma)
    evi = 2*v+(alpha)
    reg = error*evi

    return reg.mean()

def EvidentialRegression(evidential_output, y_true, coeff=1e-2):
    gamma, v, alpha, beta = evidential_output.chunk(4,1)
    df = 2*alpha
    loc = gamma
    scale = beta*(v+1)/(v*alpha)
    loss_nll = Student_NLL(y_true, df, loc, scale)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    
    return loss_nll + coeff * loss_reg


def compute_auxiliary_loss(y, pred, loss_aux_type):
    idx_seq_max, idx_seq_min = get_local_maxmin_index_sequence(y)
    y_local_max, y_local_min = y.gather(2, idx_seq_max), y.gather(2, idx_seq_min)
    pred_local_max, pred_local_min = pred.gather(2, idx_seq_max), pred.gather(2, idx_seq_min)
    loss_aux1 = compute_loss(pred_local_max, y_local_max, loss_aux_type)
    loss_aux2 = compute_loss(pred_local_min, y_local_min, loss_aux_type)

    idx_seq_max, idx_seq_min = get_local_maxmin_index_sequence(pred)
    y_local_max, y_local_min = y.gather(2, idx_seq_max), y.gather(2, idx_seq_min)
    pred_local_max, pred_local_min = pred.gather(2, idx_seq_max), pred.gather(2, idx_seq_min)
    loss_aux3 = compute_loss(pred_local_max, y_local_max, loss_aux_type)
    loss_aux4 = compute_loss(pred_local_min, y_local_min, loss_aux_type)

    return loss_aux1 + loss_aux2 + loss_aux3 + loss_aux4


def compute_loss(y, out, loss_type, zeta=None):
    L1_measurer = nn.L1Loss()
    MSE_measurer = nn.MSELoss()
    if loss_type == 'L1':
        pred = out[:, :1]
        loss = L1_measurer(pred, y)
    elif loss_type == 'MSE':
        pred = out[:, :1]
        loss = MSE_measurer(out, y)
    elif loss_type == 'evi':
        loss = EvidentialRegression(out, y, zeta)

    return loss


def performance_check(out, y, stats, n_seg=10):
    mean = stats['ABP_mean']
    std = stats['ABP_std']

    out = std * out + mean
    y = std * y + mean

    B, C, T = y.shape
    out = out.contiguous().view(B * n_seg, 1, T // n_seg)
    y = y.contiguous().view(B * n_seg, 1, T // n_seg)

    MAE = (out - y).abs().mean()
    MSE = ((out - y) ** 2).mean()
    SBP = (out.max(dim=2)[0] - y.max(dim=2)[0]).abs().mean()
    MAP = (out.mean(dim=2) - y.mean(dim=2)).abs().mean()
    DBP = (out.min(dim=2)[0] - y.min(dim=2)[0]).abs().mean()

    return MAE, MSE, SBP, MAP, DBP


def get_local_maxmin_index_sequence(y, seg_size=100):
    B, C, T = y.shape
    assert C == 1

    idx_seq_max = torch.tensor([]).view(B, C, 0).to(y.device).long()
    idx_seq_min = torch.tensor([]).view(B, C, 0).to(y.device).long()
    for i in range(0, T, seg_size):
        idx_max = y[:, :, i: i + seg_size].max(dim=2)[1].unsqueeze(1) + i
        idx_min = y[:, :, i: i + seg_size].min(dim=2)[1].unsqueeze(1) + i
        idx_seq_max = torch.cat((idx_seq_max, idx_max), dim=2)
        idx_seq_min = torch.cat((idx_seq_min, idx_min), dim=2)

    return idx_seq_max, idx_seq_min
