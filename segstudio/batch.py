"""Local manifest-driven segmentation and optional ground-truth evaluation."""

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .core import segment, mask_stats
from .evaluate import metrics, encode_rle


def run(manifest, output, backend="grabcut"):
    path = Path(manifest)
    items = json.loads(path.read_text())
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must be a nonempty list")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, item in enumerate(items):
        try:
            with Image.open(path.parent / item["image"]) as source:
                if source.width * source.height > 24_000_000:
                    raise ValueError("Image too large")
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            mask = segment(rgb, item["box"], backend)
            Image.fromarray(mask * 255).save(out / f"{i:05d}_mask.png")
            row = {
                "image": item["image"],
                "status": "ok",
                "measurements": mask_stats(mask),
                "rle": encode_rle(mask),
            }
            if "truth" in item:
                with Image.open(path.parent / item["truth"]) as source:
                    row["metrics"] = metrics(mask, np.asarray(source.convert("L")))
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            row = {"index": i, "status": "error", "error": str(exc)}
        rows.append(row)
    (out / "results.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "items": rows,
                "failures": sum(r["status"] == "error" for r in rows),
            },
            indent=2,
        )
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", default="outputs/batch")
    p.add_argument("--backend", choices=["grabcut", "sam2"], default="grabcut")
    a = p.parse_args()
    rows = run(a.manifest, a.output, a.backend)
    if any(r["status"] == "error" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
