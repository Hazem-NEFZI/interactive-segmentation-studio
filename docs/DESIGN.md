# Design: segmentation and annotation

The default backend is classical GrabCut. SAM 2.1 is optional and requires the upstream package and checkpoint. RGB input must be uint8 H×W×3; silent numeric wrapping is rejected. Rectangle coordinates are integer pixel bounds. GrabCut boxes must leave some known background outside the box.

Binary masks accept 0/1 or 0/255; area counts foreground pixels rather than summing 255-valued bytes. Evaluation reports IoU, Dice and tolerance-based boundary F1. Empty/empty masks score one; empty/nonempty masks score zero. Boundary tolerance defaults to two pixels and must be reported with results.

Annotation export uses uncompressed COCO-style RLE in column-major order. Round-trip tests verify exact reconstruction. This is a segmentation RLE object, not an entire COCO dataset with images/categories/annotations records.

The app runs inference on request and tracks the image/box/backend combination so a stale mask is not displayed after prompt changes. Large images are rejected. Ground-truth masks must match dimensions and contain binary values. No physical area is inferred without a calibration scale.

SAM integration follows the upstream predictor API, but its installed package/checkpoint combination was not exercised in this release. It currently supports box prompts only. The model is reconstructed per call; a production multi-user deployment needs controlled predictor lifecycle and concurrency management.
