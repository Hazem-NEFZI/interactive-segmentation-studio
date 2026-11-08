"""Binary segmentation metrics and lossless COCO uncompressed RLE export."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from .core import binary_mask


def metrics(prediction, truth, tolerance=2):
    a, b = binary_mask(prediction).astype(bool), binary_mask(truth).astype(bool)
    if a.shape != b.shape:
        raise ValueError("Shapes differ")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid boundary tolerance")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    total = a.sum() + b.sum()
    ea = a ^ binary_erosion(a)
    eb = b ^ binary_erosion(b)
    if not ea.any() or not eb.any():
        bf = float(not ea.any() and not eb.any())
    else:
        precision = float(np.mean(distance_transform_edt(~eb)[ea] <= tolerance))
        recall = float(np.mean(distance_transform_edt(~ea)[eb] <= tolerance))
        bf = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "iou": float(inter / union) if union else 1.0,
        "dice": float(2 * inter / total) if total else 1.0,
        "boundary_f1": bf,
    }


def encode_rle(mask):
    m = binary_mask(mask)
    flat = m.ravel(order="F")
    counts = []
    last = 0
    run = 0
    for value in flat:
        if value == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = int(value)
    counts.append(run)
    return {"size": list(m.shape), "counts": counts}


def decode_rle(rle):
    size = rle.get("size")
    counts = rle.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(not isinstance(n, int) or n <= 0 for n in size)
    ):
        raise ValueError("Invalid size")
    total = size[0] * size[1]
    if total > 24_000_000:
        raise ValueError("RLE exceeds pixel limit")
    if (
        not isinstance(counts, list)
        or any(not isinstance(n, int) or n < 0 for n in counts)
        or sum(counts) != total
    ):
        raise ValueError("Invalid RLE runs")
    return (
        np.repeat(np.arange(len(counts)) % 2, counts)
        .astype(np.uint8)
        .reshape(size, order="F")
    )
