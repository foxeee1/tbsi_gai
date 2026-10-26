from model.restoration_model import restoration_model
from data.dataset_real import DatasetDenoise_real
from data.dataset_gaussian import DatasetDenoise_gaussian
import utils.utils_image as util
from utils.utils_option import parse, save
from utils.utils_io import log

import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import multiprocessing
from torch.utils.data import DataLoader


def val_step(model, val_loader):
    ave_psnr, ave_ssim = 0, 0
    for data in val_loader:
        pred = model.validation_step(data)
        data_H = data['H'].cuda()
        ave_psnr += util.calculate_psnr(util.tensor2uint(pred), util.tensor2uint(data_H))
        ave_ssim += util.get_ssim(pred.permute(0, 2, 3, 1), data_H.permute(0, 2, 3, 1))
    return ave_psnr / len(val_loader), ave_ssim / len(val_loader)


def train_step(model, data, args):
    model.train_step(data, sup_type=args['sup_type'])

    if model.iter % args['print_every'] == 0:
        model.print_net()

    if model.iter % args['save_every'] == 0:
        model.save_net()

    if model.iter % args['validate_every'] == 0:
        ave_psnr, ave_ssim = val_step(model, val_loader)
        val_message = 'iter: %d, PSNR: %6.4f, SSIM: %6.4f' % (model.iter, ave_psnr, ave_ssim) + '\n'
        val_log = os.path.join(args['save_dir'], 'val_log.out')
        log(val_log, val_message, also_print=False)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/tbsn_unsup.json', help='Path to option JSON file.')
    opt_args = parser.parse_args()
    args = parse(opt_args.opt)
    torch.manual_seed(5)

    if args['noise_type'] == 'real':
        trainset = DatasetDenoise_real(args['n_channel'], args['train_img_size'], args['trainset_path'], 1, 'train')
        valset = DatasetDenoise_real(args['n_channel'], 0, args['valset_path'], 0, 'val')

    if args['noise_type'] == 'gaussian':
        trainset = DatasetDenoise_gaussian(args['n_channel'], args['train_img_size'], args['sigma'], args['trainset_path'], 'non-blind', 'train')
        valset = DatasetDenoise_gaussian(args['n_channel'], 0, args['sigma'], args['valset_path'], 'non-blind', 'val')

    if args['noise_type'] == 'blind_gaussian':
        trainset = DatasetDenoise_gaussian(args['n_channel'], args['train_img_size'], args['sigma'], args['trainset_path'], 'blind', 'train')
        valset = DatasetDenoise_gaussian(args['n_channel'], 0, 25, args['valset_path'], 'non-blind', 'val')

    train_loader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    model = restoration_model(args)
    model.data_parallel()

    os.makedirs(args['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(args['save_dir'], 'checkpoints'), exist_ok=True)
    save(args)

    while True:
        for i, data in enumerate(train_loader):
            train_step(model, data, args)
            if model.iter >= args['num_iters']:
                break
        if model.iter >= args['num_iters']:
            break