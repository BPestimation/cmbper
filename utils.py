import torch
import numpy as np
import os

def set_random_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)


def make_dirs(model_desc):
    def _mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    save_dir = "params/{}".format(model_desc)
    log_dir = "logs/{}".format(model_desc)
    _mkdir(save_dir)
    _mkdir(log_dir)
    return save_dir, log_dir


def save_checkpoint(epoch, global_step, model, optimizer, save_dir, name):
    checkpoint_path = os.path.join(save_dir, name)
    torch.save({"state_dict": model.state_dict(), 
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer}, checkpoint_path)


def get_model_config(args, postprocess=False):
    calibration = True if postprocess and args.loss == 'evi' else False
    use_temperature_alpha = True if calibration and args.use_temperature_alpha else False
    config = {'chin': args.chin,
              'chout': args.chout,
              'hidden': args.hidden,
              'depth': args.depth,
              'kernel_size': args.kernel_size,
              'stride': args.stride,
              'calibration': calibration,
              'use_temperature_alpha': use_temperature_alpha}
    return config


def get_checkpoint_path(args, postprocess=False):
    model_desc = get_model_description(args, postprocess)
    checkpoint_path = "params/{}/{}_model.pth".format(model_desc, args.load_type)
    return checkpoint_path


def get_model_description(args, postprocess=False):
    model_desc = "loss-{}_bsz-{}_hid-{}_dep-{}_ks-{}_st-{}_lr-{}_eta-{}".format(args.loss,
                                                            args.bsz,
                                                            args.hidden,
                                                            args.depth, 
                                                            args.kernel_size, 
                                                            args.stride, 
                                                            args.lr,
                                                            args.eta)
    if args.loss == 'evi' or (postprocess and args.loss == 'L1'):
        model_desc += '_zeta-{}'.format(args.zeta)
    if postprocess:
        model_desc += '_pp'
    return model_desc


def get_logger(log_dir):
    log_train = open(os.path.join(log_dir, 'train.txt'), 'a')
    log_valid = open(os.path.join(log_dir, 'valid.txt'), 'a')
    log_info = open(os.path.join(log_dir, 'info.txt'), 'a')
    return log_train, log_valid, log_info


def write_experiment_info(log_info, args, model):
    log_info.write('----- ARGS -----\n')
    log_info.flush()
    for key, item in vars(args).items():
        log_info.write('{} {}\n'.format(key, item))
        log_info.flush()
    log_info.write('----------------\n\n')
    log_info.flush()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info.write('{} {}\n'.format('# of params', n_params))
    log_info.flush()   
    print('# of params:', n_params)