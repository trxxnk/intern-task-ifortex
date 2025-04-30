import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings


def _load_exam_data(path: str) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла.

    :param path: Путь к CSV-файлу с данными
    :return: DataFrame с загруженными данными
    """
    return pd.read_csv(path)


def _preprocess_exam_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует категориальные данные в числовые.

    :param data: Исходный DataFrame с признаками и метками
    :return: DataFrame с заменёнными категориальными признаками
    """
    feature_replace_rules = {
        "Сон накануне": {"Да": 1, "Нет": 0},
        "Настроение": {"Хорошее": 2, "Нормальное": 1, "Плохое": 0},
        "Энергетиков накануне": {"4+": 4, "2-3": 3, "1": 1, "0": 0},
        "Посещаемость занятий": {"Высокая": 2, "Средняя": 1, "Низкая": 0},
        "Время подготовки": {
            "За неделю": 3,
            "За несколько дней": 2,
            "Последняя ночь": 1,
            "Последний час": 0,
        },
        "Сдал": {"Да": 1, "Нет": 0},
    }

    warnings.filterwarnings("ignore", category=FutureWarning)
    new_data = data.replace(feature_replace_rules)
    return new_data


def get_classification_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает и обрабатывает данные, возвращает признаки и метки.

    :param path: Путь к CSV-файлу с данными
    :return: Кортеж (X, y), где X — матрица признаков, y — массив меток
    """
    exam_data = _load_exam_data(path)
    dataset = _preprocess_exam_data(exam_data)
    X = dataset.drop(['Сдал'], axis=1).values
    y = dataset['Сдал'].values
    return X, y


def get_best_model(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    """
    Обучает модель дерева решений с подбором гиперпараметров с помощью GridSearchCV.

    :param X_train: Обучающая выборка признаков
    :param y_train: Метки обучающей выборки
    :return: Лучшая обученная модель DecisionTreeClassifier
    """
    param_grid = {
        'criterion': ['entropy'],
        'max_depth': [7, 10, 13],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [2, 5, 7]
    }

    tree = DecisionTreeClassifier()

    grid_search = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
