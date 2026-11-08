"""Local segmentation workspace. Run: python -m streamlit run app.py"""

import io, json, zipfile
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from segstudio.core import segment, overlay, mask_stats, validate_rgb
from segstudio.demo import create_sample
from segstudio.evaluate import metrics, encode_rle

st.set_page_config(page_title="Segmentation Studio", page_icon="◈", layout="wide")
st.caption("HAZEM NEFZI / COMPUTER VISION LAB")
st.title("Interactive Segmentation Studio")
st.markdown(
    "**Prompt. Inspect. Measure. Export.** A local workspace for reproducible object masks."
)
with st.sidebar:
    st.header("Input & prompts")
    uploaded = st.file_uploader("RGB image", type=["jpg", "jpeg", "png"])
    backend = st.selectbox("Backend", ["grabcut", "sam2"])
    st.caption("GrabCut: classical CPU segmentation. SAM 2: optional pretrained model.")
try:
    if uploaded:
        image = Image.open(uploaded)
        if image.width * image.height > 24_000_000:
            raise ValueError("Resize to below 24 megapixels")
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    else:
        rgb = create_sample()
    validate_rgb(rgb)
except (ValueError, OSError, Image.DecompressionBombError) as exc:
    st.error(str(exc))
    st.stop()
h, w = rgb.shape[:2]
if min(h, w) < 6:
    st.error("Use an image at least 6×6 pixels")
    st.stop()
with st.sidebar:
    # Range sliders keep a valid pair even when left/top coordinates change.
    xr = st.slider("Horizontal box", 1, w - 1, (max(1, w // 7), min(w - 1, 6 * w // 7)))
    yr = st.slider("Vertical box", 1, h - 1, (max(1, h // 7), min(h - 1, 6 * h // 7)))
    truth_file = st.file_uploader("Optional ground-truth mask", type=["png"])
    run = st.button("Segment image", type="primary", use_container_width=True)
if not uploaded and "seg_result" not in st.session_state:
    run = True
import hashlib

key = (hashlib.sha256(rgb.tobytes()).hexdigest(), xr, yr, backend)
if run:
    try:
        with st.spinner("Computing mask…"):
            mask = segment(rgb, (xr[0], yr[0], xr[1], yr[1]), backend)
        st.session_state.seg_result = (key, mask)
    except (ValueError, RuntimeError, Exception) as exc:
        st.error(str(exc))
cols = st.columns(2)
cols[0].image(rgb, caption="Source RGB", use_container_width=True)
record = st.session_state.get("seg_result")
if record and record[0] == key:
    mask = record[1]
    annotated = overlay(rgb, mask)
    stats = mask_stats(mask)
    cols[1].image(
        annotated, caption=f"{backend.upper()} mask overlay", use_container_width=True
    )
    a, b, c = st.columns(3)
    a.metric("Object area", f"{stats['area_pixels']:,} px")
    b.metric("Coverage", f"{100 * mask.mean():.1f}%")
    c.metric("Contour length", f"{stats['perimeter_pixels']:,.1f} px")
    report = {
        "backend": backend,
        "box_xyxy": [xr[0], yr[0], xr[1], yr[1]],
        "measurements": stats,
        "rle": encode_rle(mask),
    }
    if truth_file:
        try:
            gt = np.asarray(Image.open(truth_file).convert("L"))
            report["evaluation"] = metrics(mask, gt)
            st.json(report["evaluation"])
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in [("mask.png", mask * 255), ("overlay.png", annotated)]:
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            z.writestr(name, buffer.getvalue())
        z.writestr("annotation.json", json.dumps(report, indent=2))
    st.download_button(
        "Download mask + annotation",
        output.getvalue(),
        "segmentation.zip",
        "application/zip",
    )
else:
    cols[1].info("Select the object bounds, then press Segment image.")
st.caption(
    "Area is measured in pixels; physical dimensions require calibration. All processing occurs in this app runtime."
)
