## This script is adapted from "Restormer: Efficient Transformer for High-Resolution Image Restoration"
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import utils.utils_image as utils

from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from utils.utils_option import parse
from model.restoration_model import restoration_model

parser = argparse.ArgumentParser(description='Real Image Denoising on DND')

parser.add_argument('--opt', type=str, default='options/evaluate/real/eva_nlformer_dnd.json', help='Path to option JSON file.')
opt_args = parser.parse_args()
args = parse(opt_args.opt)

result_dir_mat = os.path.join(args['result_dir'], 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args['save_images']:
    result_dir_png = os.path.join(args['result_dir'], 'png')
    os.makedirs(result_dir_png, exist_ok=True)

model_restoration = restoration_model(args)
model_restoration.data_parallel()

israw = False
eval_version="1.0"

# Load info
infos = h5py.File(os.path.join(args['input_dir'], 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']

# Process data
with torch.no_grad():
    for i in tqdm(range(50)):
        Idenoised = np.zeros((20,), dtype=object)
        filename = '%04d.mat'%(i+1)
        filepath = os.path.join(args['input_dir'], 'images_srgb', filename)
        img = h5py.File(filepath, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)

        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T

        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1],idx[2]:idx[3],:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration.validation_step({'L':noisy_patch})
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            Idenoised[k] = restored_patch

            if args['save_images']:
                save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
                utils.imsave(img_as_ubyte(restored_patch), save_file)

        # save denoised data
        sio.savemat(os.path.join(result_dir_mat, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )
