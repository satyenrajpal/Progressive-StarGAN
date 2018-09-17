import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import math

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True
    
    config.log_dir= os.path.join(config.save_dir,'logs')
    config.model_save_dir= os.path.join(config.save_dir,'models')
    config.result_dir= os.path.join(config.save_dir,'results')
    config.sample_dir= os.path.join(config.save_dir,'samples')
    
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_args = None
    rafd_args = None
    celebaHQ_args= None
    affectNet_args=None

    if config.dataset in ['CelebA', 'Both']:
        celeba_args = {'dataset':config.dataset,
        'img_dir': config.celeba_image_dir,
        'attr_path': config.attr_path,
        'selected_attrs': config.selected_attrs,
        'crop_size': config.celeba_crop_size,
        'mode': config.mode,
        'num_workers': config.num_workers}
    
    if config.dataset in ['RaFD', 'Both', 'HQ']:
        rafd_args = {'dataset':config.dataset,
        'img_dir': config.rafd_image_dir,
        'attr_path': None,
        'selected_attrs': None,
        'crop_size': config.rafd_crop_size,
        'mode': config.mode,
        'num_workers': config.num_workers}
    
    if config.dataset in ['CelebA-HQ','HQ']:
        celebaHQ_args={'dataset': config.dataset,
        'h5_path':config.h5_path,
        'selected_attrs':config.selected_attrs,
        'mode':config.mode,
        'num_workers':config.num_workers,
        'hq_attr_path':config.hq_attr_path,
        'attr_path':config.attr_path}
    
    # if config.dataset in ['AffectNet','HQ']:
    #     affectNet_args={'dataset':config.dataset,
    #     'selected_attrs':config.selected_attrs,
    #     'mode':config.mode,
    #     'num_workers':4,
    #     'img_dir':config.affectNet_dir,
    #     'aNet_labels':config.aNet_labels}
    
    # Solver for training and testing StarGAN.
    solver = Solver(config,celeba_args=celeba_args, rafd_args=rafd_args,
        celebaHQ_args=celebaHQ_args)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD','CelebA-HQ']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
        elif config.dataset in ['HQ']:
            solver.train_multi_pro()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both','HQ']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=8, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=325, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_ct', type=float, default=2.0, help='weight for consistency loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both','CelebA-HQ','AffectNet','HQ'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--start_step', type=int, default=0, help='Steps to start from')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young','Eyeglasses','Mouth_Slightly_Open','Bangs'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    #       CelebA
    parser.add_argument('--celeba_image_dir', type=str, default='../CelebA_nocrop/img_celeba')
    parser.add_argument('--attr_path', type=str, default='../CelebA_nocrop/list_attr_celeba.txt')
    #       CelebA-HQ
    parser.add_argument('--h5_path', type=str, default='../CelebA-HQ/celebaHQ')
    parser.add_argument('--hq_attr_path', type=str, default='../CelebA-HQ/image_list.txt')
    #       AffectNet
    parser.add_argument('--affectNet_dir',type=str,default='affectNet')
    parser.add_argument('--aNet_labels',type=str,default='affectNet/processed_labels_train.txt')
    
    #       RaFD
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    
    parser.add_argument('--save_dir', type=str, default='hq')
    parser.add_argument('--model_restore_dir',type=str,default='hq/models')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    # print(config)
    main(config)