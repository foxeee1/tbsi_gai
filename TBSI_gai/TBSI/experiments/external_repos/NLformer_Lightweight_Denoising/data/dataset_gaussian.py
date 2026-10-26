import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetDenoise_gaussian(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, n_channels = 3, H_size = 128, sigma = 25, path = 'trainset', noise_type = 'non-blind', stage = 'train'):
        super(DatasetDenoise_gaussian, self).__init__()
        print('Synthesized Dataset: Denosing with fixed sigma.')
        self.n_channels = n_channels
        self.patch_size = H_size
        self.sigma = sigma
        self.noise_type = noise_type
        self.stage = stage

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(path)

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        """
        # --------------------------------
        # get L/H patch pairs
        # --------------------------------
        """
        H, W, _ = img_H.shape

        if self.stage == 'train':
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()
        
        elif self.stage == 'val':
            img_H = util.uint2tensor3(img_H)
            img_L = img_H.clone()

        else:
            raise ValueError('Stage should be either train or val')

        # --------------------------------
        # add noise
        # --------------------------------
        if self.noise_type == 'blind':
            # Uniformly select a noise level between self.sigma[0] and self.sigma[1]
            sigma = random.randint(self.sigma[0], self.sigma[1])
            noise = torch.randn(img_L.size()).mul_(sigma/255.0)
        else:
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)

        img_L.add_(noise)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
