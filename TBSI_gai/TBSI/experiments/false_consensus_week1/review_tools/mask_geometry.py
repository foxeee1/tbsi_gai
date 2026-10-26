#!/usr/bin/env python3
"""Convert a reviewed image-space ROI into a 16x16 search-token mask."""
import numpy as np


def roi_to_token_mask(roi_xywh, image_to_search, search_wh=(256, 256), grid_hw=(16, 16)):
    """Return a boolean mask; zero covered token centers is an explicit failure."""
    x, y, w, h = map(float, roi_xywh)
    if w <= 0 or h <= 0:
        raise ValueError("invalid reviewed ROI")
    transform = np.asarray(image_to_search, dtype=np.float64)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("image_to_search must be finite 3x3 affine matrix")
    if abs(np.linalg.det(transform)) < 1e-12:
        raise ValueError("image_to_search must be invertible")
    inverse = np.linalg.inv(transform)
    gh, gw = grid_hw
    sw, sh = search_wh
    xs = (np.arange(gw, dtype=np.float64) + 0.5) * sw / gw
    ys = (np.arange(gh, dtype=np.float64) + 0.5) * sh / gh
    sx, sy = np.meshgrid(xs, ys)
    homogeneous = np.stack((sx.ravel(), sy.ravel(), np.ones(gh * gw)))
    origin = inverse @ homogeneous
    px = origin[0] / origin[2]
    py = origin[1] / origin[2]
    mask = ((px >= x) & (px < x + w) & (py >= y) & (py < y + h))
    mask = mask.reshape(gh, gw)
    if not mask.any():
        raise ValueError("reviewed ROI covers zero token centers")
    return mask


def test_identity_and_shift():
    identity = np.eye(3)
    mask = roi_to_token_mask((0, 0, 16, 16), identity)
    assert mask.shape == (16, 16) and int(mask.sum()) == 1 and bool(mask[0, 0])
    shifted = np.array([[1, 0, -16], [0, 1, 0], [0, 0, 1]], dtype=float)
    mask = roi_to_token_mask((16, 0, 16, 16), shifted)
    assert int(mask.sum()) == 1 and bool(mask[0, 0])


if __name__ == "__main__":
    test_identity_and_shift()
    print("geometry smoke tests OK")
