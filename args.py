import argparse
import os


def parse_args():
    parser=argparse.ArgumentParser(description='Continuous monitoring of blood pressure with evidential regression.')
    parser.add_argument('--tr_path', default='datasets/train.p',type=str, help='Training data path.')
    parser.add_argument('--val_path', default='datasets/valid.p',type=str, help='Validation data path.')
    parser.add_argument('--te_path', default='datasets/test.p',type=str, help='Test data path')
    parser.add_argument('--stats_path', default='datasets/stats.p',type=str, help='Data statistics path')
    parser.add_argument('--result_path', default='results',type=str, help='Directory to save experimental results.')
    parser.add_argument('--max_itr', default=500000,type=int, help='Maximum number of iterations to train.')
    parser.add_argument('--bsz', default=512,type=int, help='Batch size.')
    parser.add_argument('--num_workers', default=4,type=int, help='Number of workers for training loader.')
    parser.add_argument('--lr', default=5e-4,type=float, help='Learning rate.')
    parser.add_argument('--chin', default=1,type=int, help='Input size')
    parser.add_argument('--chout', default=4,type=int, help='Output size')
    parser.add_argument('--hidden', default=64,type=int, help='First hidden channel size.')
    parser.add_argument('--depth', default=4,type=int, help='Number of blocks in encoder and decoder.')
    parser.add_argument('--kernel_size', default=6,type=int, help='Kernel size of convolution layer.')
    parser.add_argument('--stride', default=2,type=int, help='Stride of convolution layer.')
    parser.add_argument('--eta', default=1.0,type=float, help='Coefficient of peak-to-peak matching loss.')
    parser.add_argument('--zeta', default=0.0,type=float, help='Coefficient of penalty term in total evidential loss.')

    parser.add_argument('--loss', default='L1', choices=['L1', 'MSE', 'NLL', 'evi'], help='Main objective function.')
    parser.add_argument('--loss_aux', default='L1', choices=['L1', 'MSE'], help='Objective function of peak-to-peak matching loss.')
        
    parser.add_argument('--load_type', default=None, choices=[None, 'best_loss', 'best_MAE', 'latest'], help='Model type to load.')
    parser.add_argument('--postprocess', action="store_true", help='Whether to  load post-processed model.')
    parser.add_argument('--use_temperature_alpha', action="store_true", help='Whether to use additional temperature parameter for uncertainty calibration.')
    parser.add_argument('--post_itr', default=1e3, type=int, help='Number of iterations for post-processing.')
    parser.add_argument('--post_lr', default=5e-4,type=float, help='Learning rate for post-processing.') 

    parser.add_argument('--subset_ratio', default=0.8,type=float, help='Ratio at which high-reliability samples are selected.') 
   
    args = parser.parse_args()

    return args
