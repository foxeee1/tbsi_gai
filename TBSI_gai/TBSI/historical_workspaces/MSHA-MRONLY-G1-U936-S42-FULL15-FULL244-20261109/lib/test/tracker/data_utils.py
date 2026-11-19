import torch
import numpy as np
from lib.utils.misc import NestedTensor


class Preprocessor(object):
    """Preprocessor with cached mean/std and FP16 support."""
    _mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    _std_rgb = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    _mean_rgbt = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449]).view((1, 6, 1, 1)).cuda()
    _std_rgbt = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226]).view((1, 6, 1, 1)).cuda()
    _half_rgb = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda().half()
    _std_half_rgb = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda().half()
    _mean_half_rgbt = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449]).view((1, 6, 1, 1)).cuda().half()
    _std_half_rgbt = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226]).view((1, 6, 1, 1)).cuda().half()

    def __init__(self):
        pass

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray, half=False):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        if img_tensor.shape[1] == 3:
            mean, std = (Preprocessor._half_rgb, Preprocessor._std_half_rgb) if half else (Preprocessor._mean_rgb, Preprocessor._std_rgb)
        else:
            mean, std = (Preprocessor._mean_half_rgbt, Preprocessor._std_half_rgbt) if half else (Preprocessor._mean_rgbt, Preprocessor._std_rgbt)
        img_tensor_norm = ((img_tensor / 255.0) - mean) / std
        if half:
            img_tensor_norm = img_tensor_norm.half()
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
