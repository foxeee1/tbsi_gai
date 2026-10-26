import numpy as np
import argparse
from tqdm import tqdm
import os
import torch
import scipy.io as sio
from utils.utils_option import parse
from model.restoration_model import restoration_model

parser = argparse.ArgumentParser(description='Real Image Denoising on SIDD')

parser.add_argument('--opt', type=str, default='options/evaluate/real/eva_nlformer_sidd.json', help='Path to option JSON file.')
opt_args = parser.parse_args()
args = parse(opt_args.opt)

model_restoration = restoration_model(args)
model_restoration.data_parallel()

# Process data
filepath = 'dataset/test/SIDD/ValidationNoisyBlocksSrgb.mat'
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)

with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration.validation_step({'L':noisy_patch})
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

# save denoised data
save_dir = os.path.join("save_evaluation", "sidd")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, args['network_type'] + '_Idenoised.mat')
sio.savemat(save_path, {"Idenoised": restored,})