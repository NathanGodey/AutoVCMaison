import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn
from torch_utils import device

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.dataset + '/spmel', config.batch_size, config.len_crop)

    solver = Solver(vcc_loader, config)

    solver.train()
    solver.save_model(config.dataset + '/' + config.checkpoint)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=16)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    parser.add_argument('--init_model', type=str, default='')

    # Checkpoint path
    parser.add_argument('--checkpoint', type=str, default='autovc.ckpt')
    parser.add_argument('--checkpoint_mode', type=str, default='autosave')
    parser.add_argument('--save_every_n_iter', type=int, default=0)
    parser.add_argument('--sample_conversion_every_n_iter', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='default')


    # Training configuration.
    parser.add_argument('--dataset', type=str, default="training_set", help='dataset dir')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=10, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    print('use device: ', device)
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)
