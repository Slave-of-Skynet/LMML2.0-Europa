# United Kingdom — Call My BLEUff

Reproduce corpus BLEU scoring and the listener check from the same panel and reference files.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python solve.py
```

## Input and output

panel.csv and references.csv.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1Ycd99_NLLKqwjW80B3sca9KGNodF2pNW).

## Files

- [`bleu.py`](bleu.py)
- [`requirements.txt`](requirements.txt)
- [`solve.py`](solve.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
