import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data_manager import ARTDataset
from model import Model
from utils import get_model_config, get_model_description, make_dirs, get_logger, write_experiment_info, save_checkpoint, set_random_seed
from loss import compute_loss, compute_auxiliary_loss, performance_check
from args import parse_args
from random import sample
import numpy as np
import pickle
import os
import argparse
import json
import time


def train(epoch, writer, log_train, args):
    global global_step
    loss_all_avg = loss_main_avg = MSE_avg = MAE_avg = SBP_avg = MAP_avg = DBP_avg = 0.
    model.train()
    for i, (x, y) in enumerate(tr_loader):
        global_step += 1
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out[:, :1, :]
        loss_main = compute_loss(y, out, args.loss, args.zeta)
        loss_aux = compute_auxiliary_loss(y, pred, args.loss_aux)
        loss_all = loss_main + args.eta * loss_aux
        optimizer.zero_grad()
        loss_all.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        with torch.no_grad():
            MAE, MSE, SBP, MAP, DBP = performance_check(pred, y, stats)
        loss_all_avg += loss_all.item() / len(tr_loader)
        loss_main_avg += loss_main.item() / len(tr_loader)
        MAE_avg += MAE.item() / len(tr_loader)
        MSE_avg += MSE.item() / len(tr_loader)
        SBP_avg += SBP.item() / len(tr_loader)
        MAP_avg += MAP.item() / len(tr_loader)
        DBP_avg += DBP.item() / len(tr_loader)
        writer.add_scalar('Train/{}_loss_all'.format(args.loss), loss_all.item(), global_step)
        writer.add_scalar('Train/{}_loss_main'.format(args.loss), loss_main.item(), global_step)
        writer.add_scalar('Train/{}_loss_aux'.format(args.loss), loss_aux.item(), global_step)
        writer.add_scalar('Train/MAE', MAE.item(), global_step)
        writer.add_scalar('Train/MSE', MSE.item(), global_step)
        writer.add_scalar('Train/SBP', SBP.item(), global_step)
        writer.add_scalar('Train/MAP', MAP.item(), global_step)
        writer.add_scalar('Train/DBP', DBP.item(), global_step)

    state = {}
    state['Epoch'] = epoch
    state['Global step'] = global_step
    state['{}_loss_all'.format(args.loss)] = loss_all_avg
    state['{}_loss_main'.format(args.loss)] = loss_main_avg
    state['MAE'] = MAE_avg
    state['MSE'] = MSE_avg
    state['SBP'] = SBP_avg
    state['MAP'] = MAP_avg
    state['DBP'] = DBP_avg
    log_train.write('%s\n' % json.dumps(state))
    log_train.flush()

    print('[Train] Epoch: {}, Itr:{}, Loss: {:0.4f}, Loss-main: {:0.4f}, MAE: {:0.4f}, MSE: {:0.4f} SBP: {:0.4f}, MAP: {:0.4f}, DBP: {:0.4f}'.format(
        epoch, global_step, loss_all_avg, loss_main_avg, MAE_avg, MSE_avg, SBP_avg, MAP_avg, DBP_avg))


def validate(epoch, writer, log_valid, args):
    global global_step
    
    loss_all_avg = loss_main_avg = MSE_avg = MAE_avg = SBP_avg = MAP_avg = DBP_avg = 0.
    model.eval()
    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out[:, :1, :]
        loss_main = compute_loss(y, out, args.loss, args.zeta)
        loss_aux = compute_auxiliary_loss(y, pred, args.loss_aux)
        loss_all = loss_main + args.eta * loss_aux
        MAE, MSE, SBP, MAP, DBP = performance_check(pred, y, stats)
        loss_all_avg += loss_all.item() / len(val_loader)
        loss_main_avg += loss_main.item() / len(val_loader)
        MAE_avg += MAE.item() / len(val_loader)
        MSE_avg += MSE.item() / len(val_loader)
        SBP_avg += SBP.item() / len(val_loader)
        MAP_avg += MAP.item() / len(val_loader)
        DBP_avg += DBP.item() / len(val_loader)

    writer.add_scalar('Valid/{}_loss_main'.format(args.loss), loss_main_avg, global_step)
    writer.add_scalar('Valid/{}_loss_all'.format(args.loss), loss_all_avg, global_step)
    writer.add_scalar('Valid/MAE', MAE_avg, global_step)
    writer.add_scalar('Valid/MSE', MSE_avg, global_step)
    writer.add_scalar('Valid/SBP', SBP_avg, global_step)
    writer.add_scalar('Valid/MAP', MAP_avg, global_step)
    writer.add_scalar('Valid/DBP', DBP_avg, global_step)
    state = {}
    state['Epoch'] = epoch
    state['Global step'] = global_step
    state['{}_loss_all'.format(args.loss)] = loss_all_avg
    state['{}_loss_main'.format(args.loss)] = loss_main_avg
    state['MAE'] = MAE_avg
    state['MSE'] = MSE_avg
    state['SBP'] = SBP_avg
    state['MAP'] = MAP_avg
    state['DBP'] = DBP_avg
    log_valid.write('%s\n' % json.dumps(state))
    log_valid.flush()

    print('[Valid] Epoch: {}, Itr:{}, Loss: {:0.4f}, Loss-main: {:0.4f}, MAE: {:0.4f}, MSE: {:0.4f} SBP: {:0.4f}, MAP: {:0.4f}, DBP: {:0.4f}'.format(
        epoch, global_step, loss_all_avg, loss_main_avg, MAE_avg, MSE_avg, SBP_avg, MAP_avg, DBP_avg))

    return loss_all_avg, MAE_avg


if __name__ == "__main__":
    global global_step
    global_step = 0
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_dataset = ARTDataset(args.tr_path)
    val_dataset = ARTDataset(args.val_path)
    tr_loader = DataLoader(tr_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False)

    config = get_model_config(args)
    model = Model(**config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    stats = pickle.load(open(args.stats_path, 'rb'))
    model_desc = get_model_description(args)
    save_dir, log_dir = make_dirs(model_desc)
    log_train, log_valid, log_info = get_logger(log_dir)
    writer = SummaryWriter(logdir=os.path.join(log_dir, 'runs', str(time.strftime('%Y-%m-%d_%H:%M:%S'))))
    write_experiment_info(log_info, args, model)

    loss_best = MAE_best = 987654321
    for epoch in range(1, 987654321):
        print('# --- {}th epoch start --- # '.format(epoch))
        train(epoch, writer, log_train, args)
        with torch.no_grad():
            loss_val, MAE_val = validate(epoch, writer, log_valid, args)

        save_checkpoint(epoch, global_step, model, optimizer, save_dir, "latest_model.pth")
        print('Model Saved! - [latest_model.pth]')
        if loss_val < loss_best:
            loss_best = loss_val
            save_checkpoint(epoch, global_step, model, optimizer, save_dir, "best_loss_model.pth")
            print('Model Saved! - [best_loss_model.pth]')
        if MAE_val < MAE_best:
            MAE_best = MAE_val
            save_checkpoint(epoch, global_step, model, optimizer, save_dir, "best_MAE_model.pth")
            print('Model Saved! - [best_MAE_model.pth]')
        print('Best Valid Loss: {:0.4f}'.format(loss_best))
        print('# --- {}th epoch end --- # \n'.format(epoch))

        if global_step >= args.max_itr:
            break
    print('Done.')
