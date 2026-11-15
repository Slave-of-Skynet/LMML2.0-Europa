"""Reveal Belgium's hidden word using edge_m ~ perimeter + ditch_m."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COLUMNS = ["area_ha", "long_side_m", "ditch_m", "edge_m"]


def fit_residuals(frame):
    """Fit ordinary least squares with an intercept; keep every surveyed row."""
    missing = sorted(set(COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    values = frame[COLUMNS].to_numpy(dtype=float)
    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Expected at least four rows of finite numeric measurements")
    area_ha, long_side, ditch, edge = values.T
    if np.any(area_ha <= 0) or np.any(long_side <= 0):
        raise ValueError("area_ha and long_side_m must be positive")

    short_side = area_ha * 10_000.0 / long_side
    perimeter = 2.0 * (long_side + short_side)
    design = np.column_stack([np.ones(len(edge)), perimeter, ditch])
    coefficients, _, rank, _ = np.linalg.lstsq(design, edge, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("The regression design must have full column rank")
    residuals = edge - design @ coefficients
    total_ss = np.sum((edge - edge.mean()) ** 2)
    if total_ss == 0:
        raise ValueError("edge_m must vary to compute R^2")
    r2 = 1.0 - np.sum(residuals ** 2) / total_ss
    return perimeter, residuals, coefficients, r2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=ROOT / "potato_fields.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "residuals_plot.png")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV not found: {args.csv}. Supply the contest potato_fields.csv")

    frame = pd.read_csv(args.csv)
    perimeter, residuals, coefficients, r2 = fit_residuals(frame)
    intercept, perimeter_coef, ditch_coef = coefficients
    print(f"Rows: {len(frame)}")
    print("Model: edge_m ~ intercept + perimeter + ditch_m")
    print(f"Intercept: {intercept:.10f}")
    print(f"Perimeter coefficient: {perimeter_coef:.10f}")
    print(f"Ditch coefficient: {ditch_coef:.10f}")
    print(f"R^2: {r2:.10f}")
    print(f"Residual range (m): {residuals.min():.6f} .. {residuals.max():.6f}")

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.scatter(perimeter, residuals, s=6, c="black", linewidths=0)
    ax.set_title(f"Belgium: edge_m ~ perimeter + ditch_m (with intercept), R^2 = {r2:.7f}")
    ax.set_xlabel("Field perimeter (m)")
    ax.set_ylabel("Residual: observed edge_m - fitted edge_m (m)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"Plot: {args.output.resolve()}")
    print("Read the hidden word visually from the residuals against perimeter.")


if __name__ == "__main__":
    main()
