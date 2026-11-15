"""
Corrected evaluation pipeline for e-Residency application fraud detection.
Fixes applicant-level identity leakage via group leave-one-out cross-validation.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    csv_path = Path("applications.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "applications.csv"

    df = pd.read_csv(csv_path)

    # Shipped features and label
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

    # Applicant group identity based on date_of_birth + citizenship
    groups = df["date_of_birth"].astype(str) + "_" + df["citizenship"].astype(str)

    # Shipped model wrapped in pipeline to prevent preprocessing leakage
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=1)),
    ])

    # Leave-one-group-out cross validation (group equivalent of LeaveOneOut)
    cv = LeaveOneGroupOut()
    oof_predictions = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)

    true_accuracy = accuracy_score(y, oof_predictions)
    print(f"Corrected Leave-One-Group-Out Accuracy: {true_accuracy:.4f} ({true_accuracy * 100:.2f}%)")
    print(f"Flag: SIGMOID_{round(true_accuracy * 100)}")


if __name__ == "__main__":
    main()
