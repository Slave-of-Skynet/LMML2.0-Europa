# Germany — Black Box

Use factorial probes, mixed finite differences and boundary bisection to recover the black-box response within the query budget.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
# Evaluator API: main.solve(oracle, points).
```

## Input and output

The charged oracle and evaluation points provided by the organizer harness.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`main.py`](main.py)
- [`README.md`](README.md)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
