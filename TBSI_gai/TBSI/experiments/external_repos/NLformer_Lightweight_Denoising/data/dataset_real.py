import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import utils.utils_image as util

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, train=False):
    b,c,w,h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    unshuffled = unshuffled.view(b, c, f * f, w // f, h // f).permute(0, 2, 1, 3, 4)
    if train:
        c_num = random.randint(0, f * f - 1)
        unshuffled = unshuffled[:, c_num]
        unshuffled = unshuffled.reshape(b, c, w // f, h // f)
    else:
        unshuffled = unshuffled.reshape(b * f * f, c, w // f, h // f)
    return unshuffled

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int):
    b,c,h,w = x.shape
    before_shuffle = x.view(b // (f * f), f * f, c, h, w).permute(0, 2, 1, 3, 4)
    before_shuffle = before_shuffle.reshape(b // (f * f), c * f * f, h, w)
    return F.pixel_shuffle(before_shuffle, f)

class DatasetDenoise_real(data.Dataset):

    def __init__(self, n_channels = 3, H_size = 128, path = 'trainset', pd_factor=1, stage='train'):
        super(DatasetDenoise_real, self).__init__()
        print('Successfully build RW denoising dataset for ', stage)
        self.n_channels = n_channels
        self.patch_size = H_size
        self.stage = stage
        self.pd_factor = pd_factor # Output size is patch_size/pd_factor

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_L = util.get_image_paths(os.path.join(path, 'input_crops'))
        self.paths_H = util.get_image_paths(os.path.join(path, 'target_crops'))

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        H, W, _ = img_H.shape

        # --------------------------------
        # randomly crop the patch
        # --------------------------------
        if self.stage == 'train':
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            if self.pd_factor == 1:
                img_H = util.uint2tensor3(patch_H)
                img_L = util.uint2tensor3(patch_L)

            else:
                img_H = pixel_shuffle_down_sampling(util.uint2tensor4(patch_H), f=self.pd_factor, train=True).squeeze(0)
                img_L = pixel_shuffle_down_sampling(util.uint2tensor4(patch_L), f=self.pd_factor, train=True).squeeze(0)
        
        elif self.stage == 'val':
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)
        
        else:
            raise ValueError('Stage should be either train or val')

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
