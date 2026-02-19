"""
Утилиты для построения графиков сравнения моделей.
"""
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_model_comparison(
        metrics_path="model_metrics.json",
        output_dir="result_plots",
        figsize=(14, 9)
):
    """
    Строит графики сравнения моделей БЕЗ стратификации.

    Формат входного JSON:
    {
        'Linear': {
            'mse': {'mean': ..., 'max': ..., 'p95': ...},
            'freq_error': {'mean': ..., 'max': ..., 'p95': ...},
            ...
        },
        'RandomForest': {...},
        ...
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    # Модели и их цвета
    model_order = [
        'Linear',
        'RandomForest',
        'GradientBoosting',
        'LinearCombined',
        'GradientBoostingCombined',
        'RandomForestCombined'
    ]

    model_colors = {
        'Linear': '#5c9edc',
        'RandomForest': '#e76a6a',
        'GradientBoosting': '#6fbf6f',
        'LinearCombined': '#1f77b4',
        'GradientBoostingCombined': '#2ca02c',
        'RandomForestCombined': '#d62728'
    }

    model_labels = {
        'Linear': 'Linear',
        'RandomForest': 'RF',
        'GradientBoosting': 'GB',
        'LinearCombined': 'Linear\nCombined',
        'GradientBoostingCombined': 'GB\nCombined',
        'RandomForestCombined': 'RF\nCombined'
    }

    metrics = {
        'mse': 'RMSE, °',
        'me': 'ME, °',
        'freq_error': 'FE, %',
        'slope_error': 'SE, %'
    }

    # Фильтруем только присутствующие модели
    available_models = [m for m in model_order if m in data]

    # Презентационные шрифты
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 26,
        'legend.fontsize': 20,
        'axes.linewidth': 1.2
    })

    for metric_key, y_label in metrics.items():
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, linestyle='--', alpha=0.35, zorder=0)

        x = np.arange(len(available_models))
        width = 0.6

        means, maxs, p95s, colors, labels = [], [], [], [], []

        for model_name in available_models:
            model_data = data[model_name]

            if metric_key not in model_data:
                # Если метрики нет, пропускаем
                continue

            metric_vals = model_data[metric_key]

            mean_v = metric_vals['mean']
            max_v = metric_vals['max']
            p95_v = metric_vals.get('p95', mean_v * 1.5)  # Если p95 нет, примерная оценка

            # RMSE = sqrt(MSE)
            if metric_key == 'mse':
                mean_v = math.sqrt(mean_v)
                max_v = math.sqrt(max_v)
                p95_v = math.sqrt(p95_v)

            means.append(mean_v)
            maxs.append(max_v)
            p95s.append(p95_v)
            colors.append(model_colors.get(model_name, '#888888'))
            labels.append(model_labels.get(model_name, model_name))

        if not means:
            print(f"Пропускаем метрику {metric_key} - нет данных")
            plt.close(fig)
            continue

        y_max = max(maxs) * 1.15
        ax.set_ylim(0, y_max)

        # Для FE задаём явные тики по оси Y
        if metric_key == 'freq_error':
            max_tick = int(math.ceil(y_max))
            y_ticks = list(range(0, max_tick + 1, 1))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(tick) for tick in y_ticks])

        # Рисуем столбцы с усами
        for i, (mean_v, max_v, p95_v, color) in enumerate(
                zip(means, maxs, p95s, colors)
        ):
            left = x[i] - width / 2

            # Столбец до p95
            rect = Rectangle(
                (left, 0),
                width,
                p95_v,
                facecolor=color,
                edgecolor='black',
                linewidth=1.0,
                zorder=3
            )
            ax.add_patch(rect)

            # Линия среднего
            ax.plot(
                [left, left + width],
                [mean_v, mean_v],
                color='black',
                linewidth=2.0,
                zorder=4
            )

            # Ус до максимума
            if max_v > p95_v:
                ax.plot(
                    [x[i], x[i]],
                    [p95_v, max_v],
                    color='black',
                    linewidth=1.5,
                    zorder=4
                )
                cap = width * 0.45
                ax.plot(
                    [x[i] - cap / 2, x[i] + cap / 2],
                    [max_v, max_v],
                    color='black',
                    linewidth=1.8,
                    zorder=4
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=22)

        plt.subplots_adjust(bottom=0.18, top=0.95, left=0.12, right=0.98)
        ax.set_ylabel(y_label, fontsize=26)
        ax.xaxis.set_tick_params(pad=15)
        ax.set_title("")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_key}_comparison.svg",
                    dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/{metric_key}_comparison.pdf",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Сохранён график: {metric_key}_comparison")


def plot_metrics_table(metrics_path="model_metrics.json", output_path="result_plots/metrics_table.txt"):
    """
    Создаёт текстовую таблицу со всеми метриками.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("СРАВНЕНИЕ МЕТРИК МОДЕЛЕЙ\n")
        f.write("=" * 100 + "\n\n")

        # Заголовок таблицы
        f.write(f"{'Модель':<30} {'Mean FE %':<12} {'Max FE %':<12} {'Mean SE':<12} {'Max SE':<12} {'Mean MAE':<12}\n")
        f.write("-" * 100 + "\n")

        for model_name, metrics in data.items():
            fe_mean = metrics.get('freq_error', {}).get('mean', 0)
            fe_max = metrics.get('freq_error', {}).get('max', 0)
            se_mean = metrics.get('slope_error', {}).get('mean', 0)
            se_max = metrics.get('slope_error', {}).get('max', 0)
            mae_mean = metrics.get('mae', {}).get('mean', 0)

            f.write(f"{model_name:<30} {fe_mean:<12.4f} {fe_max:<12.4f} "
                    f"{se_mean:<12.6f} {se_max:<12.6f} {mae_mean:<12.4f}\n")

        f.write("=" * 100 + "\n")

    print(f"Таблица метрик сохранена в {output_path}")


def plot_error_distributions(models_dict, data, output_dir="result_plots"):
    """
    Строит гистограммы распределения ошибок FE и SE для всех моделей.

    Parameters:
    -----------
    models_dict : dict
        Словарь {model_name: trained_model}
    data : dict
        Словарь с данными {'x': ..., 'y': ..., 'metadata': ...}
    output_dir : str
        Директория для сохранения графиков
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    X = data['x']
    y_true = data['y']
    metadata = data['metadata']

    # Увеличенные размеры шрифтов
    font_size = 24
    label_size = 28
    tick_size = 24

    for model_name, model in models_dict.items():
        # Получаем предсказания
        y_pred = []
        for i in range(len(X)):
            a_wg, b_wg, c_wg, d_wg = y_true[i][:4]  # Берём из входных данных
            f0 = metadata[i][0]
            pred = model.predict(a_wg, b_wg, c_wg, d_wg, f0)
            y_pred.append(pred)
        y_pred = np.array(y_pred)

        # Вычисляем ошибки
        fe_errors = []
        se_errors = []

        for i in range(len(y_true)):
            f0 = metadata[i][0]
            fe = model._compute_frequency_error(y_true[i], y_pred[i], f0)
            se = model._compute_slope_error(y_true[i], y_pred[i])
            fe_errors.append(fe)
            se_errors.append(se)

        # График FE
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = label_size
        plt.rcParams['xtick.labelsize'] = tick_size
        plt.rcParams['ytick.labelsize'] = tick_size

        sns.histplot(
            fe_errors,
            bins=50,
            color='royalblue',
            alpha=0.7,
            kde=True,
            stat='count'
        )

        plt.xlabel('Ошибка частоты (FE), %')
        plt.ylabel('Количество')
        plt.title(f'{model_name} - Распределение FE', fontsize=label_size)
        plt.grid(True, alpha=0.3)

        plt.savefig(f"{output_dir}/{model_name}_fe_distribution.pdf", bbox_inches='tight')
        plt.savefig(f"{output_dir}/{model_name}_fe_distribution.svg", bbox_inches='tight')
        plt.close()

        # График SE
        plt.figure(figsize=(12, 8))
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = label_size

        sns.histplot(
            se_errors,
            bins=50,
            color='green',
            alpha=0.7,
            kde=True,
            stat='count'
        )

        plt.xlabel('Ошибка наклона (SE)')
        plt.ylabel('Количество')
        plt.title(f'{model_name} - Распределение SE', fontsize=label_size)
        plt.grid(True, alpha=0.3)

        plt.savefig(f"{output_dir}/{model_name}_se_distribution.pdf", bbox_inches='tight')
        plt.savefig(f"{output_dir}/{model_name}_se_distribution.svg", bbox_inches='tight')
        plt.close()

        print(f"Сохранены графики распределений для {model_name}")
