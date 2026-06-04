#!/bin/bash
# Export trained Path 2 model + push V154 to Kaggle.
# Run this only once `checkpoints/path2_v2/best.pt` exists.
set -e

cd "$(dirname "$0")"
PYTHON=/home/soh/miniconda3/envs/birdclef/bin/python
KAGGLE=/home/soh/miniconda3/bin/kaggle

CKPT="${1:-checkpoints/path2_v2/best.pt}"
echo "=== Exporting Path 2 ONNX from $CKPT ==="
$PYTHON export_path2.py --ckpt "$CKPT" --out kaggle_model/path2_v1.onnx

echo
echo "=== Updating kaggle_model dataset ==="
$KAGGLE datasets version -p kaggle_model -m "Add Path 2 multi-window joint model (cross-window attention) for V154"

echo
echo "=== Switching kernel-metadata.json to V154 ==="
$PYTHON - <<'PY'
import json, pathlib
p = pathlib.Path("kaggle_notebook/kernel-metadata.json")
m = json.loads(p.read_text())
m["code_file"] = "birdclef2026-v154-path2-mwjoint.py"
p.write_text(json.dumps(m, indent=2))
print("kernel-metadata.json updated:", m["code_file"])
PY

echo
echo "=== Pushing V154 notebook ==="
$KAGGLE kernels push -p kaggle_notebook

echo
echo "=== Done. Checking submission status ==="
$KAGGLE competitions submissions birdclef-2026 | head -8
