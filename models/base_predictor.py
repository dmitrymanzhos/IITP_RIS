import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, make_scorer
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
            c = np.clip(c, f0 - 0.05 * f0, f0 + 0.05 * f0)
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

    def _evaluate_curves(self, X_eval, y_eval, y_pred, metadata_eval, save_to_self=True):
        """
        УНИФИЦИРОВАННЫЙ метод вычисления метрик.

        Параметры:
        ----------
        X_eval : array-like
            Входные признаки для оценки
        y_eval : array-like
            Истинные целевые значения (коэффициенты a, b, c, d)
        y_pred : array-like
            Предсказанные целевые значения
        metadata_eval : list
            Метаданные для каждого примера (f0, H, er1, ...)
        save_to_self : bool, default=True
            Если True, сохраняет метрики в self.mean_*, self.max_*
            Если False, только возвращает словарь метрик

        Возвращает:
        -----------
        dict : Словарь с метриками
        """
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

        metrics = {
            'mean_mse': np.mean(mse_values),
            'mean_mae': np.mean(mae_values),
            'mean_me': np.mean(me_values),
            'max_mse': np.max(mse_values),
            'max_mae': np.max(mae_values),
            'max_me': np.max(me_values),
            'mean_freq_error': np.mean(freq_errors),
            'max_freq_error': np.max(freq_errors),
            'mean_slope_error': np.mean(slope_errors),
            'max_slope_error': np.max(slope_errors)
        }

        if save_to_self:
            self.mean_mse = metrics['mean_mse']
            self.mean_mae = metrics['mean_mae']
            self.mean_me = metrics['mean_me']
            self.max_mse = metrics['max_mse']
            self.max_mae = metrics['max_mae']
            self.max_me = metrics['max_me']
            self.mean_freq_error = metrics['mean_freq_error']
            self.max_freq_error = metrics['max_freq_error']
            self.mean_slope_error = metrics['mean_slope_error']
            self.max_slope_error = metrics['max_slope_error']

            if self.verbose:
                print(f"Средняя MSE: {self.mean_mse:.4f}")
                print(f"Средняя MAE: {self.mean_mae:.4f}")
                print(f"Средняя ошибка частоты (FE): {self.mean_freq_error:.2f}%")
                print(f"Средняя ошибка наклона (SE): {self.mean_slope_error:.4f}")

        return metrics

    def _create_fe_se_scorer(self):
        """Скорер, напрямую оптимизирующий FE и SE"""

        def fe_se_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            fe_errors = []
            se_errors = []

            for i in range(len(y)):
                f0 = X[i, 4]
                # Не применяем constraints во время CV — даём модели свободу
                fe = self._compute_frequency_error(y[i], y_pred[i], f0)
                se = self._compute_slope_error(y[i], y_pred[i])
                fe_errors.append(fe)
                se_errors.append(se)

            # SE переводим в % (*100) для сопоставимости с FE
            return -(np.mean(fe_errors) + np.mean(se_errors) * 100)

        return make_scorer(fe_se_scorer, greater_is_better=False)

    def predict(self, a_wg, b_wg, c_wg, d_wg, f0):
        input_data = np.array([[a_wg, b_wg, c_wg, d_wg, f0, f0 * b_wg, a_wg * b_wg]])
        pred = self.model.predict(input_data)[0]
        return self.apply_constraints(pred, f0)

    def get_metrics(self):  # возвращает метрики в виде словаря для удобной записи в файл
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

    def evaluate_overfitting(self, X_train, y_train, metadata_train):
        """
        Оценка переобучения через сравнение метрик на train и test.
        Теперь использует _evaluate_curves с save_to_self=False для train.
        """
        if self.model is None:
            print("Модель не обучена")
            return

        y_pred_train = self.model.predict(X_train)
        # Получаем метрики на train, НЕ сохраняя в self
        train_metrics = self._evaluate_curves(X_train, y_train, y_pred_train, metadata_train, save_to_self=False)

        # Метрики на test уже сохранены в self после вызова train()
        test_metrics = {
            'mean_mse': self.mean_mse,
            'mean_mae': self.mean_mae,
            'mean_me': self.mean_me,
            'mean_freq_error': self.mean_freq_error,
            'mean_slope_error': self.mean_slope_error
        }

        self._print_overfitting_report(train_metrics, test_metrics)

        if hasattr(self.model, 'estimators_'):
            self._analyze_tree_depth()

    def _print_overfitting_report(self, train_metrics, test_metrics):
        metrics_names = {
            'mean_mse': 'MSE',
            'mean_mae': 'MAE',
            'mean_me': 'Max Error',
            'mean_freq_error': 'Freq Error %',
            'mean_slope_error': 'Slope Error'
        }
        print("\n" + "=" * 80)
        print("АНАЛИЗ ПЕРЕОБУЧЕНИЯ")
        print("=" * 80)
        for metric_key, metric_name in metrics_names.items():
            train_val = train_metrics[metric_key]
            test_val = test_metrics[metric_key]
            difference = test_val - train_val
            overfitting_ratio = test_val / train_val if train_val != 0 else float('inf')
            print(f"{metric_name:15} | Обучающая: {train_val:8.4f} | Тестовая: {test_val:8.4f} | "
                  f"Разница: {difference:8.4f} | Коэффициент: {overfitting_ratio:5.2f}x")

    def _analyze_tree_depth(self):
        """Анализ глубины деревьев для нативных RF/GB (без MultiOutputRegressor)"""
        depths = []

        # Случай: Нативный RandomForestRegressor или GradientBoostingRegressor
        if hasattr(self.model, 'estimators_'):
            for tree_estimator in self.model.estimators_:
                # Для RandomForest: estimators_ это список деревьев
                if hasattr(tree_estimator, 'tree_'):
                    depths.append(tree_estimator.tree_.max_depth)
                # Для GradientBoosting: estimators_ это массив массивов деревьев
                elif hasattr(tree_estimator, '__iter__'):
                    for tree in tree_estimator:
                        if hasattr(tree, 'tree_'):
                            depths.append(tree.tree_.max_depth)

        if depths:
            print(f"\nГЛУБИНА ДЕРЕВЬЕВ:")
            print(f"  Средняя: {np.mean(depths):.1f}")
            print(f"  Максимальная: {np.max(depths)}")
            print(f"  Минимальная: {np.min(depths)}")
            print("=" * 80 + "\n")
