# -*- coding: utf-8 -*-
"""
LMML 2026 — Финляндия: "Трубы, утечки и график под ними" (Pipes, Leaks, and the Graph Beneath)
Сложность: Средний
Тема: Анализ графов (Graph Theory), Degree Centrality, очистка двуязычных топонимов.
"""

import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
import pandas as pd

# Словарь сопоставления шведских названий городов официальным финским топонимам.
# В Финляндии два государственных языка, и разные региональные офисы
# вели учет на финском и шведском языках.
SWEDISH_TO_FINNISH = {
    'ABO': 'TURKU',
    'ALAVO': 'ALAVUS',
    'BIRKALA': 'PIRKKALA',
    'BJORNEBORG': 'PORI',
    'BORGA': 'PORVOO',
    'BRAHESTAD': 'RAAHE',
    'ESBO': 'ESPOO',
    'FREDRIKSHAMN': 'HAMINA',
    'GRANKULLA': 'KAUNIAINEN',
    'HANGO': 'HANKO',
    'HELSINGFORS': 'HELSINKI',
    'HOGFORS': 'KARKKILA',
    'HYVINGE': 'HYVINKAA',
    'IDENSALMI': 'IISALMI',
    'IKALIS': 'IKAALINEN',
    'ILMOLA': 'ILMAJOKI',
    'INGA': 'INKOO',
    'JAKOBSTAD': 'PIETARSAARI',
    'KAJANA': 'KAJAANI',
    'KARLEBY': 'KOKKOLA',
    'KASKO': 'KASKINEN',
    'KERVO': 'KERAVA',
    'KEURU': 'KEURUU',
    'KIMITOON': 'KEMIONSAARI',
    'KORSHOLM': 'MUSTASAARI',
    'KRISTINESTAD': 'KRISTIINANKAUPUNKI',
    'KRONOBY': 'KRUUNUPYY',
    'KUMO': 'KOKEMAKI',
    'KYRKSLATT': 'KIRKKONUMMI',
    'LAHTIS': 'LAHTI',
    'LAPPO': 'LAPUA',
    'LARSMO': 'LUOTO',
    'LOJO': 'LOHJA',
    'LOVISA': 'LOVIISA',
    'LUNDO': 'LIETO',
    'MALAX': 'MAALAHTI',
    'NADENDAL': 'NAANTALI',
    'NYKARLEBY': 'UUSIKAARLEPYY',
    'NYSTAD': 'UUSIKAUPUNKI',
    'NARPES': 'NARPIO',
    'PARGAS': 'PARAINEN',
    'RASEBORG': 'RAASEPORI',
    'RAUMO': 'RAUMA',
    'RESO': 'RAISIO',
    'S:T KARINS': 'KAARINA',
    'S:T MICHEL': 'MIKKELI',
    'SIBBO': 'SIPOO',
    'SJUNDEA': 'SIUNTIO',
    'TAMMERFORS': 'TAMPERE',
    'TAVASTEHUS': 'HAMEENLINNA',
    'TORNEA': 'TORNIO',
    'TRASKANDA': 'JARVENPAA',
    'TUSBY': 'TUUSULA',
    'ULEABORG': 'OULU',
    'ULVSBY': 'ULVILA',
    'VANDA': 'VANTAA',
    'VASA': 'VAASA',
    'VICHTIS': 'VIHTI',
    'VILLMANSTRAND': 'LAPPEENRANTA',
    'VIRMO': 'MYNAMAKI',
    'VORA': 'VOYRI',
}


def strip_accents(text: str) -> str:
    """Удаляет скандинавские диакритические знаки (умлауты ä, ö, å)."""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def clean_town(name: str) -> str:
    """Очищает строку: убирает пробелы, точки, умлауты и переводит шведские названия в финские."""
    s = str(name).strip().rstrip('.').strip()
    norm = strip_accents(s).upper()
    canonical = SWEDISH_TO_FINNISH.get(norm, norm)
    return canonical.title()


def find_routes_file() -> Path:
    """Ищет routes.csv в текущей директории, рядом со скриптом или в папке проекта."""
    search_paths = [
        Path("routes.csv"),
        Path(__file__).resolve().parent / "routes.csv",
    ]
    for path in search_paths:
        if path.exists():
            return path
    raise FileNotFoundError("Файл routes.csv не найден. Укажите путь к нему.")


def main():
    routes_path = find_routes_file()
    print(f"Загрузка данных из: {routes_path}")
    df = pd.read_csv(routes_path)

    # Построение множества уникальных ребер (канонический неориентированный граф)
    edges = set()
    for _, row in df.iterrows():
        u = clean_town(row['from_town'])
        v = clean_town(row['to_town'])
        if u and v and u != v:
            # Сортируем вершины для исключения дублирования A-B и B-A
            edges.add(tuple(sorted([u, v])))

    # Сохранение очищенного файла связей
    clean_df = pd.DataFrame(list(edges), columns=['town_a', 'town_b'])
    clean_df = clean_df.sort_values(by=['town_a', 'town_b'])
    out_path = Path(__file__).resolve().parent / "routes_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Очищенная сеть сохранена в: {out_path}")

    # Построение списка смежности
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Сортировка городов по степени связности (число уникальных соседей)
    degrees = sorted([(len(adj[c]), c) for c in adj], key=lambda x: (-x[0], x[1]))

    num_towns = len(adj)
    num_connections = len(edges)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ГРАФА СЕТИ")
    print("=" * 60)
    print(f"Уникальных городов (вершин): {num_towns}")
    print(f"Уникальных маршрутов (ребер): {num_connections}")

    # Самопроверка по условию задачи
    assert num_towns == 116, f"Ошибка: ожидалось 116 городов, получено {num_towns}"
    assert num_connections == 448, f"Ошибка: ожидалось 448 соединений, получено {num_connections}"

    top8_degrees = [deg for deg, _ in degrees[:8]]
    assert len(top8_degrees) == len(set(top8_degrees)), "Ошибка: степени у топ-8 городов не уникальны!"

    print("\nТОП-8 САМЫХ ЗАГРУЖЕННЫХ ГОРОДОВ-ХАБОВ:")
    print("-" * 60)
    letters = []
    for rank, (deg, town) in enumerate(degrees[:8], 1):
        letter = town[0].upper()
        if rank <= 7:
            letters.append(letter)
            print(f"{rank}. {town:18s} | Связей (degree): {deg:2d} -> Буква: {letter}")
        else:
            print(f"{rank}. {town:18s} | Связей (degree): {deg:2d} (проверочный)")

    word = "".join(letters)
    flag = f"SIGMOID_{word}"
    print("-" * 60)
    print(f"Слово из первых букв топ-7 хабов: {word}")
    print(f"ФЛАГ ДЛЯ ПЛАТФОРМЫ: {flag}")
    print("=" * 60)


if __name__ == "__main__":
    main()
