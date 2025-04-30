import sys
import traceback

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from utils import get_classification_data, get_best_model



def print_status(message: str):
    """Печатает статусное сообщение в консоль с обрамлением."""
    print(f"\n{'='*10} {message} {'='*10}")


def main(filepath: str):
    """
    Основная функция загрузки данных, обучения модели и вывода точности.

    :param filepath: Путь к CSV-файлу с данными экзаменов
    """
    print_status("Запуск программы")

    try:
        print_status("Загрузка и предобработка данных")
        X, y = get_classification_data(filepath)

        print_status("Генерация полиномиальных признаков")
        poly = PolynomialFeatures(degree=5)
        X_poly = poly.fit_transform(X)

        print_status("Разделение данных")
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        print_status("Обучение модели")
        model = get_best_model(X_train, y_train)

        print_status("Оценка точности")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nТочность модели на тестовой выборке: {accuracy:.2f}")
        
        cross_validation = cross_val_score(model, X_poly, y, cv=5)
        print("Кросс-валидация (полученные точности): " \
              f"{[round(score, 2) for score in cross_validation]}")
        print(f"Средняя точность на кросс-валидации: {cross_validation.mean():.2f}")

    except FileNotFoundError:
        print(f"[ОШИБКА] Файл '{filepath}' не найден. Проверьте путь и попробуйте снова.")
    except Exception as e:
        print(f"[ОШИБКА] Произошла непредвиденная ошибка: {e}")
        traceback.print_exc()

    print_status("Завершение программы")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python3 main.py \"your_filename.csv\"")
    else:
        main(sys.argv[1])
