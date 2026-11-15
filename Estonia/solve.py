# -*- coding: utf-8 -*-
"""
Auditing the e-Residency fraud detection pipeline (Estonia - 'This Data Is Perfect!').

Evaluates:
1. Baseline shipped pipeline score (93.42%).
2. Decoy 1: Global StandardScaler leakage before split.
3. Decoy 2: Train/test split variance vs standard Leave-One-Out CV.
4. Real Bug: Repeated applicant leakage (group leakage) across multiple submissions.
5. Corrected pipeline and true accuracy calculation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset() -> pd.DataFrame:
    candidates = [
        Path("applications.csv"),
        Path(__file__).resolve().parent / "applications.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Could not find applications.csv")


def main():
    df = load_dataset()
    features = [
        "form_fill_seconds",
        "doc_scan_quality",
        "photo_match_score",
        "email_domain_age_days",
        "ip_geo_distance_km",
        "address_history_years",
    ]
    X = df[features]
    y = df["label"]
    applicant_ids = df["date_of_birth"].astype(str) + "_" + df["citizenship"].astype(str)

    print(f"Dataset loaded: {len(df)} rows, {applicant_ids.nunique()} unique applicants.")

    # 1. Shipped Pipeline (Reproducing 93.42%)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1137
    )
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_tr, y_tr)
    shipped_acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"[Shipped Baseline] Test accuracy: {shipped_acc:.4f} ({shipped_acc * 100:.2f}%)")

    # 2. Decoy 1: Fitting StandardScaler on full X vs only on training folds
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=1)),
    ])
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
        X, y, test_size=0.2, random_state=1137
    )
    pipe.fit(X_tr_raw, y_tr_raw)
    decoy1_acc = accuracy_score(y_te_raw, pipe.predict(X_te_raw))
    print(f"[Decoy 1 Fixed] Proper pipeline scaling on split: {decoy1_acc:.4f} ({decoy1_acc * 100:.2f}%)")

    # 3. Decoy 2: Evaluating with standard Leave-One-Out CV on full dataset
    loo = LeaveOneOut()
    preds_decoy2 = cross_val_predict(pipe, X, y, cv=loo, n_jobs=-1)
    decoy2_acc = accuracy_score(y, preds_decoy2)
    print(f"[Decoy 2 Fixed] Standard LeaveOneOut CV on full dataset: {decoy2_acc:.4f} ({decoy2_acc * 100:.2f}%)")

    # 4. Real Bug Fix: LeaveOneGroupOut CV grouped by applicant identity
    logo = LeaveOneGroupOut()
    corrected_preds = cross_val_predict(pipe, X, y, cv=logo, groups=applicant_ids, n_jobs=-1)
    true_acc = accuracy_score(y, corrected_preds)

    print("\n--- RESULTS & AUDIT SUMMARY ---")
    print(f"Reported score:     {shipped_acc:.4f} (~93.42%)")
    print(f"Decoy 1 score:      {decoy1_acc:.4f} (~93.67%) [no effect]")
    print(f"Decoy 2 score:      {decoy2_acc:.4f} (~95.94%) [no effect / slight increase]")
    print(f"True score (LOGO):  {true_acc:.4f} ({true_acc * 100:.2f}%)")

    flag_percent = round(true_acc * 100)
    print(f"\nFLAG (rounded): SIGMOID_{flag_percent}")


if __name__ == "__main__":
    main()
