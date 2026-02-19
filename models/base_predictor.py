import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
import matplotlib.pyplot as plt
import seaborn as sns
import os


class BasePredictor:  # родительский класс для всех моделей, для единообразия моделей
    def __init__(self, verbose=True, show_plots=False, random_state=42):
        self.verbose = verbose
        self.show_plots = show_plots
        self.random_state = random_state
        self.model = None
        self.x = None
        self.y = None
        self.metadata = None
        self.mean_mae = None
        self.mean_me = None
        self.mean_mse = None
        self.max_mae = None
        self.max_me = None
        self.max_mse = None
        self.mean_freq_error = None
        self.max_freq_error = None
        self.mean_slope_error = None
        self.max_slope_error = None

    def load_data(self, data):
        self.x = data['x']
        self.y = data['y']
        self.metadata = data['metadata']
        if self.verbose:
            print(f"Загружено данных: {len(self.x)} samples")

    def arctg_func(self, coefs, x):
        a, b, c, d = coefs
        return a * np.arctan(b * (x - c)) + d

    def apply_constraints(self, coeffs, f0=None):  # ограничения на коэффициенты
        a, b, c, d = coeffs
        if a >= -80:
            a = -80.01
        if b <= 0:
            b = 0.01
        if f0 is not None:
            c = np.clip(c, f0 - 0.1 * f0, f0 + 0.1 * f0)
        return [a, b, c, d]

    def _compute_frequency_error(self, true_coeffs, pred_coeffs, f0):  # вычисление FE
        a_true, b_true, c_true, d_true = true_coeffs
        a_pred, b_pred, c_pred, d_pred = pred_coeffs
        try:
            x0_true = c_true + np.tan(-d_true / a_true) / b_true
        except:
            x0_true = c_true
        try:
            x0_pred = c_pred + np.tan(-d_pred / a_pred) / b_pred
        except:
            x0_pred = c_pred
        return (abs(x0_true - x0_pred) / f0) * 100

    def _compute_slope_error(self, true_coeffs, pred_coeffs):  # вычисление SE
        a_true, b_true, _, _ = true_coeffs
        a_pred, b_pred, _, _ = pred_coeffs
        slope_true = a_true * b_true
        slope_pred = a_pred * b_pred
        if abs(slope_true) < 1e-10:
            return abs(slope_pred)
        else:
            return abs(slope_true - slope_pred) / abs(slope_true)

    def _compute_metrics(self, X, y_true, y_pred, metadata):  # возвращает все метрики в 1 словаре
        mse_v, mae_v, me_v, fe_v, se_v = [], [], [], [], []
        for i in range(len(y_true)):
            f0 = metadata[i][0]
            x_grid = np.linspace(f0 - 0.1 * f0, f0 + 0.1 * f0, 1000)
            tc = self.arctg_func(y_true[i], x_grid)
            pc = self.arctg_func(self.apply_constraints(y_pred[i], f0), x_grid)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
            mse_v.append(mean_squared_error(tc, pc))
            mae_v.append(mean_absolute_error(tc, pc))
            me_v.append(max_error(tc, pc))
            fe_v.append(self._compute_frequency_error(y_true[i], y_pred[i], f0))
            se_v.append(self._compute_slope_error(y_true[i], y_pred[i]))
        return {
            'mean_mse': np.mean(mse_v), 'mean_mae': np.mean(mae_v),
            'mean_me': np.mean(me_v), 'mean_freq_error': np.mean(fe_v),
            'mean_slope_error': np.mean(se_v)
        }

    def _evaluate_curves(self, X_eval, y_eval, y_pred, metadata_eval):  # вычисление метрик
        mse_values = []
        mae_values = []
        me_values = []
        freq_errors = []
        slope_errors = []

        for i in range(len(X_eval)):
            f0 = metadata_eval[i][0]
            x_grid = np.linspace(f0 - 0.1 * f0, f0 + 0.1 * f0, 1000)
            true_curve = self.arctg_func(y_eval[i], x_grid)
            pred_curve = self.arctg_func(self.apply_constraints(y_pred[i], f0), x_grid)

            mse_values.append(mean_squared_error(true_curve, pred_curve))
            mae_values.append(mean_absolute_error(true_curve, pred_curve))
            me_values.append(max_error(true_curve, pred_curve))
            freq_errors.append(self._compute_frequency_error(y_eval[i], y_pred[i], f0))
            slope_errors.append(self._compute_slope_error(y_eval[i], y_pred[i]))

        self.mean_mse = np.mean(mse_values)
        self.mean_mae = np.mean(mae_values)
        self.mean_me = np.mean(me_values)
        self.max_mse = np.max(mse_values)
        self.max_mae = np.max(mae_values)
        self.max_me = np.max(me_values)
        self.mean_freq_error = np.mean(freq_errors)
        self.max_freq_error = np.max(freq_errors)
        self.mean_slope_error = np.mean(slope_errors)
        self.max_slope_error = np.max(slope_errors)

        if self.verbose:
            print(f"Средняя MSE: {self.mean_mse:.4f}")
            print(f"Средняя MAE: {self.mean_mae:.4f}")
            print(f"Средняя ошибка частоты: {self.mean_freq_error:.2f}%")

    def predict(self, a_wg, b_wg, c_wg, d_wg, f0):
        input_data = np.array([[a_wg, b_wg, c_wg, d_wg, f0, f0 * b_wg, a_wg * b_wg]])
        pred = self.model.predict(input_data)[0]
        return self.apply_constraints(pred, f0)

    def get_metrics(self):  # возвращает метрики в вде словаря для удобной записи в файл
        return {
            'mean_mse': self.mean_mse,
            'mean_mae': self.mean_mae,
            'mean_me': self.mean_me,
            'max_mse': self.max_mse,
            'max_mae': self.max_mae,
            'max_me': self.max_me,
            'mean_freq_error': self.mean_freq_error,
            'max_freq_error': self.max_freq_error,
            'mean_slope_error': self.mean_slope_error,
            'max_slope_error': self.max_slope_error
        }

    def evaluate_overfitting(self, X_train, y_train, metadata_train):  # оценка переобучения
        if self.model is None:
            print("Модель не обучена")
            return
        y_pred_train = self.model.predict(X_train)
        train_metrics = self._compute_metrics(X_train, y_train, y_pred_train, metadata_train)
        test_metrics = {
            'mean_mse': self.mean_mse,
            'mean_mae': self.mean_mae,
            'mean_me': self.mean_me,
            'mean_freq_error': self.mean_freq_error,
            'mean_slope_error': self.mean_slope_error
        }
        self._print_overfitting_report(train_metrics, test_metrics)
        if hasattr(self, 'model') and hasattr(self.model, 'estimators_'):
            self._analyze_tree_depth()

    def _print_overfitting_report(self, train_metrics, test_metrics):
        metrics_names = {
            'mean_mse': 'MSE',
            'mean_mae': 'MAE',
            'mean_me': 'Max Error',
            'mean_freq_error': 'Freq Error %',
            'mean_slope_error': 'Slope Error'
        }
        for metric_key, metric_name in metrics_names.items():
            train_val = train_metrics[metric_key]
            test_val = test_metrics[metric_key]
            difference = test_val - train_val
            overfitting_ratio = test_val / train_val if train_val != 0 else float('inf')
            print(f"{metric_name:15} | Обучающая: {train_val:8.4f} | Тестовая: {test_val:8.4f} | "
                  f"Разница: {difference:8.4f} | Коэффициент: {overfitting_ratio:5.2f}x")

    def _analyze_tree_depth(self):
        if not hasattr(self.model, 'estimators_'):
            return
        depths = []
        for output_estimator in self.model.estimators_:  # по числу выходов
            if hasattr(output_estimator, 'estimators_'):  # RF внутри MultiOutput
                for tree in output_estimator.estimators_:
                    depths.append(tree.tree_.max_depth)
        if depths:
            print(f"  Средняя: {np.mean(depths):.1f}, Макс: {np.max(depths)}, Мин: {np.min(depths)}")
