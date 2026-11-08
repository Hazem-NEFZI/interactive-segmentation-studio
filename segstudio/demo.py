import argparse, json
from pathlib import Path
import cv2
import numpy as np
from .core import grabcut_mask, overlay, mask_stats


def create_sample():
    rng = np.random.default_rng(21)
    h, w = 360, 560
    bg = np.full((h, w, 3), [31, 42, 59], np.uint8)
    noise = rng.normal(0, 3, bg.shape)
    bg = np.clip(bg.astype(float) + noise, 0, 255).astype(np.uint8)
    cv2.ellipse(bg, (287, 186), (180, 82), -13, 0, 360, (232, 158, 57), -1)
    cv2.ellipse(bg, (287, 186), (150, 56), -13, 0, 360, (240, 183, 87), -1)
    cv2.circle(bg, (260, 177), 20, (57, 65, 76), -1)
    cv2.circle(bg, (260, 177), 13, (240, 183, 87), -1)
    cv2.line(bg, (175, 240), (390, 124), (70, 83, 112), 6, cv2.LINE_AA)
    return bg


def main(output="outputs"):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    rgb = create_sample()
    box = (85, 55, 480, 308)
    mask = grabcut_mask(rgb, box)
    cv2.imwrite(str(folder / "input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder / "mask.png"), mask * 255)
    cv2.imwrite(
        str(folder / "demo.png"), cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR)
    )
    result = {"backend": "grabcut", "sample": "synthetic", **mask_stats(mask)}
    (folder / "measurements.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs")
    a = p.parse_args()
    main(a.output)
