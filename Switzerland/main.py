import os
import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def extract_avalanche_flag(input_path: str = 'test-input.csv', output_path: str = 'test-output.txt') -> str:
    """
    Обучает дерево решений на наблюдениях снежных ям (WSL SLF Davos)
    и извлекает скрытое слово из порядка признаков, по которым дерево задает вопросы
    для определения схода лавины (слабый слой).
    """
    df = pd.read_csv(input_path)

    target_col = 'slab_released' if 'slab_released' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # Обучение дерева решений с параметрами по умолчанию
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    tree = clf.tree_

    # Поиск листового узла, предсказывающего сход пласта (class 1)
    target_leaf = None
    for i in range(tree.node_count):
        if tree.children_left[i] == -1 and tree.children_right[i] == -1:
            val = tree.value[i][0]
            if len(val) > 1 and val[1] > val[0]:
                target_leaf = i
                break

    # Функция поиска пути от корня дерева к целевому листу
    def find_path(node, target, path):
        if node == target:
            return path
        if tree.children_left[node] == -1 and tree.children_right[node] == -1:
            return None
        left_search = find_path(tree.children_left[node], target, path + [node])
        if left_search is not None:
            return left_search
        right_search = find_path(tree.children_right[node], target, path + [node])
        if right_search is not None:
            return right_search
        return None

    path = find_path(0, target_leaf, []) if target_leaf is not None else None

    # Извлечение первых букв признаков вдоль цепочки условий
    if path:
        word = ''.join(feature_names[tree.feature[n]][0].upper() for n in path)
    else:
        # Резервный вариант: сортировка внутренних узлов по глубине дерева
        depths = {}
        def compute_depth(node, d):
            if node == -1:
                return
            depths[node] = d
            compute_depth(tree.children_left[node], d + 1)
            compute_depth(tree.children_right[node], d + 1)

        compute_depth(0, 0)
        internal_nodes = [i for i in range(tree.node_count) if tree.children_left[i] != -1 or tree.children_right[i] != -1]
        internal_nodes.sort(key=lambda n: depths[n])
        word = ''.join(feature_names[tree.feature[n]][0].upper() for n in internal_nodes)

    flag = f'SIGMOID_{word}'

    # Запись в test-output.txt (строго одна строка, без кавычек)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(flag + '\n')

    print(f'Extracted word: {word}')
    print(f'Flag: {flag}')
    return flag


if __name__ == '__main__':
    extract_avalanche_flag('test-input.csv', 'test-output.txt')
