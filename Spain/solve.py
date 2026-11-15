"""
Task: Spain - "¡Feliz Navidad!"
Category: Easy (100 points) | Track: Europe (Data Science & Classical ML)
Author: denis100strike

Description:
Reconstructs the HR Christmas Board 2025 by auditing receipts of 21 charter pilots.
Filters valid purchases within the official fair price range for 6 Spanish regional delicacies.
Visualizes the resulting 6x21 pilot-souvenir board, revealing the hidden word: cRuCiO.
Flag: [OMITTED]
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    candidates_guide = [
        "office_price_guide.csv",
        "data/office_price_guide.csv",
    ]
    candidates_purchases = [
        "purchases.csv",
        "data/purchases.csv",
    ]

    guide_path = next((p for p in candidates_guide if os.path.exists(p)), None)
    purchases_path = next((p for p in candidates_purchases if os.path.exists(p)), None)

    if not guide_path or not purchases_path:
        raise FileNotFoundError("Could not find office_price_guide.csv or purchases.csv")

    guide = pd.read_csv(guide_path)
    purchases = pd.read_csv(purchases_path)
    return guide, purchases


def solve():
    print("[1/4] Loading Spanish souvenirs price guide and receipts...")
    guide, purchases = load_data()
    print(f"      Loaded guide ({len(guide)} products) and {len(purchases)} purchase receipts.")

    print("[2/4] Calculating unit price and applying fair price filter...")
    purchases['unit_price'] = purchases['Total_paid_eur'] / purchases['Quantity']
    df = purchases.merge(guide, on='Product')

    # Office rule: fair_min <= unit_price <= fair_max
    valid_mask = (df['unit_price'] >= df['Fair_min_eur']) & (df['unit_price'] <= df['Fair_max_eur'])
    valid = df[valid_mask]
    print(f"      Verified {len(valid)} valid purchases out of {len(df)} total.")

    print("[3/4] Rebuilding the 6x21 Christmas Board matrix...")
    grid = np.zeros((6, 21), dtype=int)
    for _, row in valid.iterrows():
        pilot_idx = int(row['Pilot_no']) - 1  # 0 to 20 (Pilots 1..21)
        list_idx = int(row['List_no']) - 1    # 0 to 5 (Products 1..6)
        grid[list_idx, pilot_idx] = 1

    print("[4/4] Generating high-resolution board visualization...")
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    ax.imshow(grid, cmap='Greens', aspect='auto', interpolation='nearest', vmin=0, vmax=1)

    ax.set_xticks(range(21))
    ax.set_xticklabels([f'P{i}' for i in range(1, 22)], fontsize=11, fontweight='bold')

    product_labels = [f'{row.List_no}. {row.Product}' for _, row in guide.sort_values('List_no').iterrows()]
    ax.set_yticks(range(6))
    ax.set_yticklabels(product_labels, fontsize=11, fontweight='bold')

    ax.set_title('Christmas Board 2025: Fair Price Souvenirs by Pilot', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Pilots (Columns 1 to 21)', fontsize=12, labelpad=10)
    ax.set_ylabel('Souvenirs (List_no 1 to 6)', fontsize=12, labelpad=10)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, 21, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which='minor', color='#999999', linestyle='-', linewidth=1.5)

    plot_path = "board_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"      Board plot saved to {plot_path}")

    print("Read the word visually from the board.")


if __name__ == "__main__":
    solve()
