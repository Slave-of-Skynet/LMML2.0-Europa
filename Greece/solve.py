# -*- coding: utf-8 -*-
"""
Решение задачи 'Just a siblings' thing' / 'Отношения братьев и сестер' (Греция)
Оценка трех моделей оракулов (Ares, Athena, Apollo) при пороге 0.50.
Сравнение метрик качества (Accuracy, Recall, Precision, F1, F2) для критической
системы предупреждения катастрофических угроз в святилище Дельф.
"""

import os
import zipfile
import numpy as np
import pandas as pd

# 9 числовых признаков согласно model.py
FEATURE_COLUMNS = [
    "creature_mass_talents", "oracle_omen", "aggression_index", "divine_aura",
    "aegis_resonance", "armor_resistance", "witness_count",
    "styx_disturbance", "omen_intensity"
]


def load_dataset():
    """
    Загружает тестовые признаки и метки классов.
    Поддерживает как официальные имена test_dataset.csv / test_labels.csv,
    так и резервные варианты file1.csv / file2.csv.
    """
    candidates = [
        (
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset.csv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_labels.csv")
        ),
        ("test_dataset.csv", "test_labels.csv"),
        (
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "file1.csv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "file2.csv")
        ),
        ("file1.csv", "file2.csv"),
    ]

    for ds_path, lb_path in candidates:
        if os.path.exists(ds_path) and os.path.exists(lb_path):
            df_features = pd.read_csv(ds_path)
            df_labels = pd.read_csv(lb_path)
            return pd.merge(df_features, df_labels, on="encounter_id")

    raise FileNotFoundError("Не удалось найти файлы датасета (test_dataset.csv / test_labels.csv).")


def find_model_path(filename):
    """Находит файл весов модели в доступных каталогах."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename),
        os.path.join("models", filename),
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join("models", filename)


def load_model_weights(pth_path, model_name):
    """
    Загружает веса модели.
    Поддерживает PyTorch (если установлен) или прямое чтение буфера из zipfile.
    """
    try:
        import torch
        state = torch.load(pth_path, map_location="cpu", weights_only=True)
        if "linear.weight" in state:
            w = state["linear.weight"].squeeze().detach().cpu().numpy().astype(np.float32)
            b = state["linear.bias"].detach().cpu().numpy().astype(np.float32)
            return w, b
    except Exception:
        pass

    # Резервный кроссплатформенный парсер: веса PyTorch .pth хранятся внутри zip-контейнера
    with zipfile.ZipFile(pth_path, 'r') as z:
        with z.open(f"{model_name}/data/0") as f:
            w = np.frombuffer(f.read(), dtype=np.float32)
        with z.open(f"{model_name}/data/1") as f:
            b = np.frombuffer(f.read(), dtype=np.float32)
    return w, b


def evaluate_models():
    data = load_dataset()
    X = data[FEATURE_COLUMNS].values
    y = data["is_catastrophic_threat"].values

    total_samples = len(y)
    total_threats = int(y.sum())
    total_safe = total_samples - total_threats

    print("=" * 68)
    print("   ОРАКУЛЫ ДЕЛЬФ: СРАВНЕНИЕ МОДЕЛЕЙ ДЛЯ ЗАЩИТЫ ОТ МИФИЧЕСКИХ УГРОЗ")
    print("=" * 68)
    print(f"Всего наблюдений в тесте:              {total_samples}")
    print(f"Истинных катастрофических угроз (y=1): {total_threats} ({total_threats / total_samples * 100:.1f}%)")
    print(f"Обычных / мирных существ (y=0):        {total_safe} ({total_safe / total_samples * 100:.1f}%)")
    print(f"Порог классификации (Threshold):        0.50 (sigmoid >= 0.50 <=> logit >= 0.0)")
    print("-" * 68)

    models = {
        "model_ares": ("ARES", "Ares (Арес)", "model_ares.pth"),
        "model_athena": ("ATHENA", "Athena (Афина)", "model_athena.pth"),
        "model_apollo": ("APOLLO", "Apollo (Аполлон)", "model_apollo.pth")
    }

    results = []

    for mod_id, (key, display_name, filename) in models.items():
        pth_path = find_model_path(filename)
        w, b = load_model_weights(pth_path, mod_id)

        # Вычисление логитов: logit = X @ w + b
        logits = X @ w + b
        # Классификация при пороге 0.50 (sigmoid(logits) >= 0.50 эквивалентно logits >= 0)
        preds = (logits >= 0.0).astype(int)

        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        tn = int(np.sum((preds == 0) & (y == 0)))

        acc = (tp + tn) / total_samples
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0

        results.append({
            "key": key,
            "name": display_name,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "specificity": spec,
            "f1": f1,
            "f2": f2
        })

        print(f"[{display_name}]")
        print(f"  Confusion Matrix: TP={tp:2d}, FP={fp:2d}, FN={fn:2d}, TN={tn:3d}")
        print(f"  Accuracy (Общая точность):        {acc * 100:.2f}%")
        print(f"  Recall (Полнота по угрозам):       {rec * 100:.2f}%   <-- РЕШАЮЩАЯ МЕТРИКА")
        print(f"  Precision (Точность детектора):   {prec * 100:.2f}%")
        print(f"  F1-Score:                         {f1:.4f}")
        print(f"  F2-Score (Recall weight = 2x):    {f2:.4f}")
        print(f"  ПРОПУЩЕНО КАТАСТРОФ (FN):         {fn:2d} из {total_threats} ({fn / total_threats * 100:.1f}%)")
        print(f"  Ложных тревог (FP):               {fp:2d} из {total_safe}")
        print("-" * 68)

    best_model = max(results, key=lambda r: r["recall"])

    print("ИТОГИ И ВЫВОД ДЛЯ ЖЮРИ:")
    print("1. 'Парадокс точности' (Accuracy Paradox):")
    print("   Все три модели имеют одинаковую точность Accuracy = 95.00% (ровно по 50 ошибок).")
    print("2. В условиях задачи пропуск катастрофической угрозы (False Negative) смертельно опасен,")
    print("   в то время как ложная тревога (False Positive) влечет лишь проверку стражей.")
    print("3. Сравнение пропусков (False Negatives):")
    print("   - Apollo:  пропустил 45 катастроф из 100 (Recall = 55.00%) -> КАТАСТРОФА ДЛЯ ДЕЛЬФ")
    print("   - Athena:  пропустила 25 катастроф из 100 (Recall = 75.00%) -> СЛИШКОМ МНОГО ЖЕРТВ")
    print("   - Ares:    пропустил всего 5 катастроф из 100 (Recall = 95.00%) -> МАКСИМАЛЬНАЯ ЗАЩИТА")
    print(f"\nМодель, выбранная для охраны святилища: {best_model['name']}")
    print(f"\nФЛАГ ДЛЯ ОТПРАВКИ НА ПЛАТФОРМУ:")
    print(f"SIGMOID_{best_model['key']}")
    print("=" * 68)


if __name__ == "__main__":
    evaluate_models()
