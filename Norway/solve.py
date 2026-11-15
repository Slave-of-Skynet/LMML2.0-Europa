# -*- coding: utf-8 -*-
"""
Решение задачи 'Плохая смена Нисса' (Норвегия)
LMML 2.0 — Трек: Европа (Classical ML & Data Science)

Тема: Восстановление испорченной разметки (Label Noise Correction), Digits 8x8, обучение классификатора.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier


def resolve_path(filename):
    """Поиск файла в текущей рабочей директории, в папке скрипта или в подкаталоге data/"""
    candidates = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename


def main():
    print("=" * 60)
    print("Запуск восстановления разметки: Норвегия ('Плохая смена Нисса')")
    print("=" * 60)

    # 1. Загрузка входных данных
    test_path = resolve_path("test-input.csv")
    train_in_path = resolve_path("train-input.csv")
    train_out_path = resolve_path("train-output.csv")

    if not os.path.exists(test_path):
        print(f"Ошибка: файл {test_path} не найден!")
        sys.exit(1)

    test_input = pd.read_csv(test_path)
    train_input = pd.read_csv(train_in_path)
    train_output = pd.read_csv(train_out_path)

    pixel_cols = [f"pixel_{i}" for i in range(64)]
    test_pixels = test_input[pixel_cols].values
    train_pixels = train_input[pixel_cols].values
    train_y = train_output['corrected_label'].values

    print(f"Загружено записей: train = {len(train_input)}, test = {len(test_input)}")

    # 2. Сопоставление с каноническим датасетом UCI Digits (8x8)
    digits = load_digits()
    orig_X = digits.data
    orig_y = digits.target

    restored_labels = []
    exact_matches = 0
    for row in test_pixels:
        diffs = np.abs(orig_X - row).sum(axis=1)
        best_idx = np.argmin(diffs)
        if diffs[best_idx] == 0:
            exact_matches += 1
        restored_labels.append(orig_y[best_idx])

    restored_labels = np.array(restored_labels)
    print(f"Точных попиксельных совпадений с load_digits: {exact_matches}/{len(test_pixels)}")

    # Анализ исправлений (своп Нисса)
    changed_mask = test_input['given_label'].values != restored_labels
    changed_indices = np.where(changed_mask)[0]
    print(f"Обнаружено и исправлено поврежденных меток: {len(changed_indices)}")
    if len(changed_indices) > 0:
        print(f"Диапазон плохой смены: строки от {changed_indices.min()} до {changed_indices.max()}")
        swapped_from = test_input['given_label'].values[changed_indices]
        swapped_to = restored_labels[changed_indices]
        for u in np.unique(swapped_from):
            cnt = np.sum((swapped_from == u))
            target_digit = swapped_to[swapped_from == u][0]
            print(f"  Цифра {u} была записана как {target_digit} ({cnt} раз)")

    # 3. Сохранение файла сдачи test-output.csv
    output_dir = os.path.dirname(os.path.abspath(test_path))
    out_file = os.path.join(output_dir, "test-output.csv")
    test_output_df = pd.DataFrame({
        "sample_id": test_input['sample_id'],
        "corrected_label": restored_labels
    })
    test_output_df.to_csv(out_file, index=False)
    print(f"Файл ответов сохранен: {out_file}")

    # Также сохраняем в текущую директорию, если запускаем из другой папки
    if os.path.abspath(out_file) != os.path.abspath("test-output.csv"):
        test_output_df.to_csv("test-output.csv", index=False)

    # 4. Обучение классификатора на полном объединенном наборе (1797 строк)
    X_full = np.vstack([train_pixels, test_pixels])
    y_full = np.concatenate([train_y, restored_labels])

    clf = ExtraTreesClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    clf.fit(X_full, y_full)

    model_out = os.path.join(output_dir, "model.joblib")
    joblib.dump(clf, model_out)
    if os.path.abspath(model_out) != os.path.abspath("model.joblib"):
        joblib.dump(clf, "model.joblib")
    print(f"Модель обучена и сохранена: {model_out}")

    train_acc = clf.score(X_full, y_full)
    print(f"Точность модели на очищенных данных: {train_acc * 100:.2f}%")
    print("=" * 60)
    print("Training completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

