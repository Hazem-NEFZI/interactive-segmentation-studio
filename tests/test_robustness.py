import numpy as np
import pytest
from segstudio.core import mask_stats, segment, overlay
from segstudio.evaluate import encode_rle, decode_rle, metrics


@pytest.mark.parametrize("seed", range(5))
def test_lossless_rle(seed):
    mask = (np.random.default_rng(seed).random((29, 37)) > 0.6).astype(np.uint8)
    assert np.array_equal(mask, decode_rle(encode_rle(mask)))
    assert metrics(mask, mask) == {"iou": 1.0, "dice": 1.0, "boundary_f1": 1.0}


def test_255_mask_area():
    m = np.zeros((10, 10), np.uint8)
    m[2:5, 3:7] = 255
    assert mask_stats(m)["area_pixels"] == 12


@pytest.mark.parametrize(
    "bad", [np.zeros((9, 9)), np.zeros((9, 9, 3), float), np.zeros((2, 2, 3), np.uint8)]
)
def test_invalid_images(bad):
    with pytest.raises(ValueError):
        segment(bad, (1, 1, 7, 7))


def test_empty_masks_and_disjoint():
    z = np.zeros((10, 10), np.uint8)
    a = z.copy()
    b = z.copy()
    a[1, 1] = 1
    b[8, 8] = 1
    assert metrics(z, z)["iou"] == 1
    assert metrics(a, b)["iou"] == 0
    assert metrics(a, b)["boundary_f1"] == 0


def test_invalid_rle_and_overlay():
    with pytest.raises(ValueError):
        decode_rle({"size": [10, 10], "counts": [1, 2]})
    with pytest.raises(ValueError):
        overlay(np.zeros((10, 10, 3), np.uint8), np.zeros((2, 2)))
