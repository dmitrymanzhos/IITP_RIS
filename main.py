import os
import numpy as np
import json
import re
import matplotlib

matplotlib.use('Agg')

from models.linear_model import LinearPredictor
from models.random_forest_model import RandomForestPredictor
from models.gradient_boosting_model import GradientBoostingPredictor
from models.linear_combined_model import LinearCombinedPredictor
from models.gradient_boosting_combined_model import GradientBoostingCombinedPredictor
from models.random_forest_combined_model import RandomForestCombinedPredictor

from utils import save_detailed_metrics, plot_model_comparison, plot_metrics_table


def load_data(file_paths, normalize_by_f0=True):
    all_data = []
    pattern = r'f=([\d.]+)GHz\s*er1=([\d.]+)\s*H=([\d.]+)\s*w=([\d.]+)\s*A=([\d.]+)\s*L=([\d.]+)'

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Предупреждение: файл {file_path} не найден, пропускаем")
            continue

        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                param_str = ' '.join(parts[:6])
                match = re.search(pattern, param_str)
                if not match:
                    continue
                try:
                    ff = float(match.group(1))
                    er1 = float(match.group(2))
                    h = float(match.group(3))
                    w = float(match.group(4))
                    a = float(match.group(5))
                    l = float(match.group(6))
                    coeffs = list(map(float, parts[6:14]))
                    all_data.append((coeffs, [ff, h, er1, w, a, l]))
                except Exception as e:
                    print(f"Ошибка обработки строки в файле {file_path}: {e}")

    if not all_data:
        raise ValueError("Не удалось загрузить данные из файлов!")

    features = []
    targets = []
    metadata = []

    for item in all_data:
        a_wg, b_wg, c_wg, d_wg = item[0][:4]
        f0 = item[1][0]

        if normalize_by_f0:  # TODO: Нормализация признаков по f0 - нужна ли
            features.append([
                a_wg,
                b_wg * f0,
                c_wg / f0,
                d_wg / abs(a_wg) if abs(a_wg) > 1e-10 else 0,
                f0,
                a_wg * b_wg,
                (c_wg - f0) / f0
            ])
        else:
            features.append([
                a_wg, b_wg, c_wg, d_wg, f0,
                f0 * b_wg,
                a_wg * b_wg
            ])

        targets.append(item[0][4:8])  # a_fs, b_fs, c_fs, d_fs
        metadata.append(item[1])

    print(f"\nЗагружено {len(features)} примеров")
    if normalize_by_f0:
        print("Признаки нормализованы по f0")

    return {
        'x': np.array(features),
        'y': np.array(targets),
        'metadata': metadata
    }


def main():
    data_files = [
        "approximated_data/ph_post_processed_data_new1_f=1GHz_sorted.txt",
        "approximated_data/ph_post_processed_data_new1_f=5GHz_sorted.txt",
        "approximated_data/ph_post_processed_data_new1_f=10GHz_sorted.txt",
        "approximated_data/ph_post_processed_data_new1_f=20GHz_sorted.txt",
        "approximated_data/ph_post_processed_data_new1_f=60GHz_sorted.txt"
    ]  # предобработанный датасет

    data = load_data(data_files, normalize_by_f0=True)

    models = {
        'Linear': LinearPredictor(verbose=True, show_plots=False),
        'RandomForest': RandomForestPredictor(verbose=True, show_plots=False),
        'GradientBoosting': GradientBoostingPredictor(verbose=True, show_plots=False),
        'LinearCombined': LinearCombinedPredictor(verbose=True, show_plots=False),
        'GradientBoostingCombined': GradientBoostingCombinedPredictor(verbose=True, show_plots=False),
        'RandomForestCombined': RandomForestCombinedPredictor(verbose=True, show_plots=False)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"МОДЕЛЬ: {name}")
        print(f"{'=' * 80}\n")

        try:
            model.load_data(data)
            if 'Combined' in name:  # для двухэтапных (сombined) моделей задаём параметры для обоих этапов
                if name == 'LinearCombined':
                    model.train(n_iter_slope=30, n_iter_coeff=30)
                elif name == 'GradientBoostingCombined':
                    model.train(n_iter_slope=50, n_iter_coeff=50)
                else:  # RandomForestCombined
                    model.train(n_iter=50)
            else:
                model.train()

            results[name] = model.get_metrics()
            trained_models[name] = model

            print(f"\n{'=' * 80}")
            print(f"РЕЗУЛЬТАТЫ МОДЕЛИ: {name}")
            print(f"{'=' * 80}")
            print(f"Mean FE: {results[name]['mean_freq_error']:.4f}%")
            print(f"Max FE:  {results[name]['max_freq_error']:.4f}%")
            print(f"Mean SE: {results[name]['mean_slope_error']:.6f}")
            print(f"Max SE:  {results[name]['max_slope_error']:.6f}")
            print(f"Mean MAE: {results[name]['mean_mae']:.4f}")
            print(f"{'=' * 80}\n")

        except Exception as e:
            print(f"Ошибка при обучении модели {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("Ни одна модель не обучилась успешно!")
        return

    output_dir = 'results_json'
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ МЕТРИК")
    print("=" * 80)
    save_detailed_metrics(trained_models, f"{output_dir}/detailed_metrics.json")

    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 80)
    plot_model_comparison(
        metrics_path=f"{output_dir}/detailed_metrics.json",
        output_dir="result_plots"
    )

    plot_metrics_table(
        metrics_path=f"{output_dir}/detailed_metrics.json",
        output_path="result_plots/metrics_table.txt"
    )

    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"{'Модель':<30} {'Mean FE %':<12} {'Max FE %':<12} {'Mean SE':<12} {'Max SE':<12}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['mean_freq_error']:<12.4f} {metrics['max_freq_error']:<12.4f} "
              f"{metrics['mean_slope_error']:<12.6f} {metrics['max_slope_error']:<12.6f}")
    print("=" * 80 + "\n")

    # Находим лучшую модель по Mean FE и SE
    best_model_fe = min(results.items(), key=lambda x: x[1]['mean_freq_error'])
    best_model_se = min(results.items(), key=lambda x: x[1]['mean_slope_error'])

    print(f"🏆 Лучшая модель по Mean FE: {best_model_fe[0]} ({best_model_fe[1]['mean_freq_error']:.4f}%)")
    print(f"🏆 Лучшая модель по Mean SE: {best_model_se[0]} ({best_model_se[1]['mean_slope_error']:.6f})")

    print(f"\n✅ Результаты сохранены в {output_dir}/")
    print(f"✅ Графики сохранены в result_plots/")


if __name__ == "__main__":
    main()
