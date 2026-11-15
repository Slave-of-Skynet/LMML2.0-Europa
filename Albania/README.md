# Albania — Looks like an anomaly!

A bounded streaming detector scores contextual, point and collective sensor anomalies.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
# Evaluator API: import Detector from detector.py; call detector.step(reading).
```

## Input and output

Streaming sensor readings supplied by the evaluator.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/14AMHetQATgeB-wSm0-Da10NffyxOGQVo).

## Files

- [`detector.py`](detector.py)
- [`README.md`](README.md)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
