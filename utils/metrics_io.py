import json
import numpy as np


def convert_numpy_to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def save_metrics(metrics, filename="model_metrics.json"):
    """
    Сохраняет метрики в JSON-формате с поддержкой numpy типов.

    Parameters:
    -----------
    metrics : dict
        Словарь с метриками моделей
    filename : str
        Путь к файлу для сохранения

    Example:
    --------
    metrics = {
        'Linear': {
            'mean_mse': 0.1234,
            'mean_mae': 0.0567,
            ...
        },
        'RandomForest': {...}
    }
    save_metrics(metrics, 'results/model_metrics.json')
    """
    # Конвертируем все numpy типы
    metrics_clean = convert_numpy_to_python(metrics)

    with open(filename, 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    print(f"Метрики сохранены в {filename}")


def load_metrics(filename="model_metrics.json"):
    """
    Загружает метрики из JSON-файла.

    Parameters:
    -----------
    filename : str
        Путь к файлу с метриками

    Returns:
    --------
    dict : Словарь с метриками

    Example:
    --------
    metrics = load_metrics('results/model_metrics.json')
    print(metrics['Linear']['mean_freq_error'])
    """
    with open(filename, 'r') as f:
        return json.load(f)


def save_detailed_metrics(models_dict, filename="detailed_metrics.json"):
    """
    Сохраняет детализированные метрики всех моделей.

    Формат для совместимости с графиками:
    {
        'model_name': {
            'mse': {'mean': ..., 'max': ..., 'p95': ...},
            'mae': {'mean': ..., 'max': ..., 'p95': ...},
            'me': {'mean': ..., 'max': ..., 'p95': ...},
            'freq_error': {'mean': ..., 'max': ..., 'p95': ...},
            'slope_error': {'mean': ..., 'max': ..., 'p95': ...}
        }
    }

    Parameters:
    -----------
    models_dict : dict
        Словарь {model_name: model_instance}
    filename : str
        Путь к файлу для сохранения
    """
    detailed = {}

    for model_name, model in models_dict.items():
        metrics = model.get_metrics()

        detailed[model_name] = {
            'mse': {
                'mean': metrics['mean_mse'],
                'max': metrics['max_mse'],
                'p95': metrics['mean_mse'] * 1.5  # Примерная оценка p95 TODO: исправить
            },
            'mae': {
                'mean': metrics['mean_mae'],
                'max': metrics['max_mae'],
                'p95': metrics['mean_mae'] * 1.5
            },
            'me': {
                'mean': metrics['mean_me'],
                'max': metrics['max_me'],
                'p95': metrics['mean_me'] * 1.5
            },
            'freq_error': {
                'mean': metrics['mean_freq_error'],
                'max': metrics['max_freq_error'],
                'p95': metrics['mean_freq_error'] * 1.5
            },
            'slope_error': {
                'mean': metrics['mean_slope_error'],
                'max': metrics['max_slope_error'],
                'p95': metrics['mean_slope_error'] * 1.5
            }
        }

    # Конвертируем и сохраняем
    detailed_clean = convert_numpy_to_python(detailed)

    with open(filename, 'w') as f:
        json.dump(detailed_clean, f, indent=2)

    print(f"Детализированные метрики сохранены в {filename}")
