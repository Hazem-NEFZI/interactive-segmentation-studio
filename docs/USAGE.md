# Usage

```bash
python -m pip install -e ".[dev,app]"
python -m streamlit run app.py
python -m segstudio.batch assets/manifest.example.json --output outputs/batch
```

Batch manifests are JSON lists with `image`, integer `box: [x0,y0,x1,y1]`, and optional `truth`. Paths resolve relative to the manifest. Every item receives a status; the CLI exits nonzero if any item fails while retaining successful outputs.

The bundled mask is a previous GrabCut result, so manifest evaluation on that mask is a reproducibility check, not an independent segmentation-accuracy experiment.

Optional SAM setup: install from https://github.com/facebookresearch/sam2 and obtain the official SAM 2.1 tiny checkpoint. Set `SAM2_CHECKPOINT`, `SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml`, and `SAM2_DEVICE=cpu` or `cuda`. On PowerShell use `$env:SAM2_CHECKPOINT="C:\path\model.pt"`. No checkpoint is downloaded automatically.
