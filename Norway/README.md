# Norway — The Nisse's Bad Shift

Restore digit labels by matching canonical 8x8 digit pixels, then train the classifier on corrected labels.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python solve.py
```

## Input and output

train-input.csv, train-output.csv and test-input.csv. The packaged model and corrected test labels are retained.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1wNDRDlEVopfH3p72PN973rtvle6-Gbo9).

## Files

- [`model.joblib`](model.joblib)
- [`README.md`](README.md)
- [`requirements.txt`](requirements.txt)
- [`solve.py`](solve.py)
- [`test-output.csv`](test-output.csv)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
