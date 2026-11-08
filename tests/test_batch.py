import json
from pathlib import Path
from segstudio.batch import run


def test_manifest_batch_and_failure(tmp_path):
    assets = Path(__file__).resolve().parents[1] / "assets"
    records = run(assets / "manifest.example.json", tmp_path / "good")
    assert records[0]["status"] == "ok"
    assert records[0]["metrics"]["iou"] > 0.98
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"image": "missing.png", "box": [1, 1, 4, 4]}]))
    assert run(bad, tmp_path / "bad")[0]["status"] == "error"
