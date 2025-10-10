import os
import numpy as np
import json
import re
import matplotlib

matplotlib.use('Agg')

from models.base_predictor import BasePredictor
from models.linear_model import LinearPredictor
from models.random_forest_model import RandomForestPredictor
from models.gradient_boosting_model import GradientBoostingPredictor
from models.random_forest_combined_model import RandomForestCombinedPredictor


def load_data(file_paths):  # загрузка датасета из предобработанных файлов
    all_data = []
    pattern = r'f=([\d.]+)GHz\s*er1=([\d.]+)\s*H=([\d.]+)\s*w=([\d.]+)\s*A=([\d.]+)\s*L=([\d.]+)'

    for file_path in file_paths:
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
    features = []
    targets = []
    metadata = []

    for item in all_data:
        a_wg, b_wg, c_wg, d_wg = item[0][:4]
        f0 = item[1][0]
        features.append([a_wg, b_wg, c_wg, d_wg, f0, f0 * b_wg, a_wg * b_wg])
        targets.append(item[0][4:8])
        metadata.append(item[1])

    return {
        'x': np.array(features),
        'y': np.array(targets),
        'metadata': metadata
    }


def main():
    data_files = [
        "ph_post_processed_data_new1_f=1GHz_sorted.txt",
        "ph_post_processed_data_new1_f=5GHz_sorted.txt",
        "ph_post_processed_data_new1_f=10GHz_sorted.txt",
        "ph_post_processed_data_new1_f=20GHz_sorted.txt",
        "ph_post_processed_data_new1_f=60GHz_sorted.txt"
    ]

    data = load_data(data_files)

    models = {
        'Linear': LinearPredictor(verbose=True, show_plots=False),
        'RandomForest': RandomForestPredictor(verbose=True, show_plots=False),
        'GradientBoosting': GradientBoostingPredictor(verbose=True, show_plots=False),
        'RandomForestCombined': RandomForestCombinedPredictor(verbose=True, show_plots=False)
    }

    results = {}
    for name, model in models.items():
        model.load_data(data)
        model.train()
        results[name] = model.get_metrics()


if __name__ == "__main__":
    main()
