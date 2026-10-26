"""Deterministic input perturbations for controlled RGB-T tracker diagnostics."""
import cv2
import numpy as np


MODES = ('rgb_failure', 'tir_failure', 'hot_background',
         'local_misalignment', 'modality_conflict')


def apply_input_perturbation(image, mode):
    if not mode:
        return image
    if mode not in MODES:
        raise ValueError('unsupported TBSI_PERTURBATION: {}'.format(mode))
    if image.ndim != 3 or image.shape[2] != 6:
        raise ValueError('RGB-T perturbation expects HxWx6 input, got {}'.format(image.shape))

    result = image.copy()
    height, width = result.shape[:2]
    if mode == 'rgb_failure':
        result[:, :, :3] = cv2.GaussianBlur(result[:, :, :3], (0, 0), sigmaX=12.0)
    elif mode == 'tir_failure':
        result[:, :, 3:] = 127
    elif mode == 'hot_background':
        center = (int(round(width * 0.82)), int(round(height * 0.18)))
        axes = (max(2, int(round(width * 0.07))), max(2, int(round(height * 0.07))))
        thermal = result[:, :, 3:].copy()
        cv2.ellipse(thermal, center, axes, 0, 0, 360, (255, 255, 255), -1)
        result[:, :, 3:] = thermal
    elif mode == 'local_misalignment':
        matrix = np.asarray([[1.0, 0.0, 8.0], [0.0, 1.0, 8.0]], dtype=np.float32)
        result[:, :, 3:] = cv2.warpAffine(
            result[:, :, 3:], matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127))
    elif mode == 'modality_conflict':
        result[:, :, 3:] = 255 - result[:, :, 3:]
    return result
