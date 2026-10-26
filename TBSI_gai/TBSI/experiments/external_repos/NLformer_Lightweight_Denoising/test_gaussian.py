from model.restoration_model import restoration_model
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

def infer(model, data, ratio, use_ensemble=True):
    if use_ensemble:
        return infer_with_ensemble(model, data, ratio)

    return model.validation_step({'L': data['L'].cuda()}, ratio)

def infer_with_ensemble(model, data, ratio):
    data_L = data['L'].cuda()
    pred = torch.zeros_like(data_L).detach()

    pred += model.validation_step({'L': data_L}, ratio)
    pred += torch.rot90(model.validation_step({'L': torch.rot90(data_L, 1, [2,3])}, ratio), 3, [2,3])
    pred += torch.rot90(model.validation_step({'L': torch.rot90(data_L, 2, [2,3])}, ratio), 2, [2,3])
    pred += torch.rot90(model.validation_step({'L': torch.rot90(data_L, 3, [2,3])}, ratio), 1, [2,3])
    pred += torch.flip(model.validation_step({'L': torch.flip(data_L, [2])}, ratio), [2])
    pred += torch.flip(torch.rot90(model.validation_step({'L': torch.rot90(torch.flip(data_L, [2]), 1, [2,3])}, ratio), 3, [2,3]), [2])
    pred += torch.flip(torch.rot90(model.validation_step({'L': torch.rot90(torch.flip(data_L, [2]), 2, [2,3])}, ratio), 2, [2,3]), [2])
    pred += torch.flip(torch.rot90(model.validation_step({'L': torch.rot90(torch.flip(data_L, [2]), 3, [2,3])}, ratio), 1, [2,3]), [2])

    return pred / 8.0

def test_gaussian(model, test_loader, save_path, log_path, ratio=128, use_ensemble=True):
    ave_psnr, ave_ssim = 0, 0
    for i, data in enumerate(test_loader):
        pred = infer(model, data, ratio, use_ensemble)
        data_H = data['H'].cuda()
        ave_psnr += 10 * torch.log10(1 / nn.functional.mse_loss(torch.clamp(pred, 0, 1), data_H)).item()
        ave_ssim += util.calculate_ssim(util.tensor2uint(pred), util.tensor2uint(data_H))

        util.imsave(util.tensor2uint(pred), os.path.join(save_path, '%04d.png' % i))
    ave_psnr /= len(test_loader)
    ave_ssim /= len(test_loader)
    log_message = 'PSNR: %6.4f, SSIM: %6.4f' % (ave_psnr, ave_ssim) + '\n'
    log(log_path, log_message, also_print=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/evaluate/gaussian/eva_nlformer_gauss_c.json', help='Path to option JSON file.')
    parser.add_argument('--use_ensemble', dest='use_ensemble', action='store_true', help='Use test-time ensemble inference.')
    parser.set_defaults(use_ensemble=True)
    opt_args = parser.parse_args()
    args = parse(opt_args.opt)

    if args['n_channel'] == 1:
        testset_names = ['BSD68', 'Set12', 'Urban100']
    elif args['n_channel'] == 3:
        testset_names = ['Kodak', 'McMaster', 'CBSD68', 'Urban100']
    else:
        raise ValueError('Invalid number of channels.')
    
    testset_path = "dataset/test"

    model = restoration_model(args)
    model.data_parallel()

    os.makedirs(args['save_dir'], exist_ok=True)
    result_log = os.path.join(args['save_dir'], 'result_log.log')

    for cur_test in testset_names:
        testset = DatasetDenoise_gaussian(args['n_channel'], 0, args['sigma'], os.path.join(testset_path, cur_test), 'non-blind', 'val')
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

        cur_save_path = os.path.join(args['save_dir'], cur_test)
        os.makedirs(cur_save_path, exist_ok=True)
        log_message = 'Testset: ' + cur_test + '\n'
        log(result_log, log_message, also_print=False)

        test_gaussian(model, test_loader, cur_save_path, result_log, use_ensemble=opt_args.use_ensemble)
