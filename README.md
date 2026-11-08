# Interactive segmentation studio

**Hazem Nefzi · Robotics / AI / Computer Vision · v2.0**

Box-prompted segmentation, binary-mask measurements, lossless annotation export, and optional SAM 2.1 integration.

![Working demonstration](assets/demo.png)

## Start in three commands

Use an isolated Python environment for each repository. From this directory:

```bash
python -m pip install -e ".[dev]"
python -m segstudio.demo --output outputs
python -m pytest -q
```

For Windows setup, see [the installation guide](docs/INSTALL.md). The package can be imported from outside the repository after installation.

## What this repository demonstrates

Box-prompted segmentation, binary-mask measurements, lossless annotation export, and optional SAM 2.1 integration. This is an engineering portfolio implementation with explicit failure handling and reproducible experiments. Sample results are synthetic; no external benchmark ranking is claimed.

| Workflow | Command / action |
|---|---|
| Basic demonstration | `python -m segstudio.demo --output outputs` |
| Extended experiment | `python -m streamlit run app.py` |
| Own data / integration | `Use the app to upload an image and optionally a binary ground-truth mask.` |
| Static checks | `python -m ruff check .` |
| Regression tests | `python -m pytest -q` |

## Evidence and engineering notes

- [Validation report](docs/VALIDATION.md): executed tests, environment, and limitations.
- [Design decisions](docs/DESIGN.md): algorithms, assumptions, and failure modes.
- [Detailed usage](docs/USAGE.md): extended workflows and input contracts.
- [Model / system card](docs/SYSTEM_CARD.md): intended use and evaluation limits.
- [Contributing](CONTRIBUTING.md): development and review expectations.

## Scope and attribution

This repository was developed with AI assistance and should be reviewed and understood before use. The original synthetic examples are independent of confidential internship or research work. They are not a release of a deployed industrial system. Optional external models and datasets retain their own licenses.

MIT license applies to this repository's original code. See `LICENSE`.
