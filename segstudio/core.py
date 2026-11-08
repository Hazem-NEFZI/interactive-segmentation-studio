import os
import cv2
import numpy as np


def validate_rgb(rgb):
    rgb = np.asarray(rgb)
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) < 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError("Image must be HxWx3 uint8 RGB with dimensions >=3")
    if rgb.shape[0] * rgb.shape[1] > 24_000_000:
        raise ValueError("Image exceeds 24 megapixel limit")
    return np.ascontiguousarray(rgb)


def binary_mask(mask):
    mask = np.asarray(mask)
    if (
        mask.ndim != 2
        or not np.isfinite(mask).all()
        or not np.isin(mask, [0, 1, 255]).all()
    ):
        raise ValueError("Mask must be a finite 2D binary image (0/1 or 0/255)")
    return (mask > 0).astype(np.uint8)


def validate_box(box, shape):
    box = np.asarray(box)
    if (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or not np.equal(box, np.floor(box)).all()
    ):
        raise ValueError("Box requires four finite integer pixel coordinates")
    x0, y0, x1, y1 = map(int, box)
    h, w = shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            "Box must satisfy 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height"
        )
    return x0, y0, x1, y1


def grabcut_mask(rgb, box, iterations=6):
    rgb = validate_rgb(rgb)
    x0, y0, x1, y1 = validate_box(box, rgb.shape)
    h, w = rgb.shape[:2]
    if not isinstance(iterations, int) or not 1 <= iterations <= 50:
        raise ValueError("Iterations must be 1..50")
    if x0 == 0 or y0 == 0 or x1 == w or y1 == h:
        raise ValueError(
            "GrabCut requires a rectangle strictly inside the image boundary"
        )
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        img, mask, (x0, y0, x1 - x0, y1 - y0), bg, fg, iterations, cv2.GC_INIT_WITH_RECT
    )
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)


def sam2_mask(rgb, box):
    rgb = validate_rgb(rgb)
    validate_box(box, rgb.shape)
    path = os.getenv("SAM2_CHECKPOINT")
    config = os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
    device = os.getenv("SAM2_DEVICE", "cpu")
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "Set SAM2_CHECKPOINT to a real downloaded SAM 2.1 checkpoint file"
        )
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "Install Meta SAM 2 from https://github.com/facebookresearch/sam2"
        ) from exc
    model = build_sam2(config, path, device=device)
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode():
        predictor.set_image(np.asarray(rgb, dtype=np.uint8))
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=False
        )
    return np.asarray(masks[np.argmax(scores)], dtype=np.uint8)


def segment(rgb, box, backend="grabcut"):
    if backend == "grabcut":
        return grabcut_mask(rgb, box)
    if backend == "sam2":
        return sam2_mask(rgb, box)
    raise ValueError("backend must be grabcut or sam2")


def overlay(rgb, mask):
    img = validate_rgb(rgb).copy()
    positive = binary_mask(mask).astype(bool)
    if positive.shape != img.shape[:2]:
        raise ValueError("Image and mask shapes differ")
    img[positive] = np.clip(
        0.52 * img[positive] + 0.48 * np.array([24, 220, 190]), 0, 255
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        positive.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return cv2.drawContours(img, contours, -1, (255, 255, 255), 2)


def mask_stats(mask):
    mask = binary_mask(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"area_pixels": 0, "bbox_xyxy": None, "perimeter_pixels": 0.0}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    return {
        "area_pixels": int(mask.sum()),
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ],
        "perimeter_pixels": round(perimeter, 2),
    }
