import numpy as np
import pytest
from segstudio.demo import create_sample
from segstudio.core import segment, mask_stats, overlay


def test_demo_grabcut():
    im = create_sample()
    mask = segment(im, (85, 55, 480, 308))
    assert mask.shape == im.shape[:2]
    assert 1000 < mask_stats(mask)["area_pixels"] < im.shape[0] * im.shape[1]
    assert overlay(im, mask).shape == im.shape


def test_box_validation():
    with pytest.raises(ValueError):
        segment(np.zeros((30, 30, 3), np.uint8), (0, 0, 30, 30))
