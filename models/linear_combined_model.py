import numpy as np
import math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, make_scorer
from .base_predictor import BasePredictor


class LinearCombinedPredictor(BasePredictor):
    """
    Двухстадийная Combined модель с линейными регрессорами.

    АРХИТЕКТУРА:
    Стадия 1 (slope_predictor): Linear Ridge → (f0, slope)
    Стадия 2 (coeff_predictor): Linear Ridge → (a, d)

    Полностью аналогична RandomForestCombinedPredictor, но использует
    линейную регрессию вместо случайного леса. Быстрее, проще,
    но может быть менее точной на нелинейных зависимостях.
    """

    def __init__(self, verbose=True, show_plots=False, random_state=42):
        super().__init__(verbose, show_plots, random_state)
        self.slope_predictor = None
        self.coeff_predictor = None

    def _prepare_slope_data(self, y_data):
        """Подготовка целевых значений для первой модели (f0, slope)"""
        y_slope = []
        for coeffs in y_data:
            f0, slope = self._compute_f0_slope(coeffs)
            y_slope.append([f0, slope])
        return np.array(y_slope)

    def _prepare_coeff_data(self, x_data, slope_predictions, y_data):
        """Подготовка данных для второй модели (X + slope → a, d)"""
        x_coeff = np.hstack([x_data, slope_predictions])
        y_coeff = y_data[:, [0, 3]]  # a и d
        return x_coeff, y_coeff

    def _compute_f0_slope(self, coeffs):
        """Вычисление f0 (резонансная частота) и slope (угол наклона в градусах)"""
        a, b, c, d = coeffs
        f0 = c
        derivative = a * b
        angle_rad = math.atan(abs(derivative))
        slope = angle_rad * (180 / math.pi)
        return f0, slope

    def _compute_b_from_slope(self, a_pred, slope_pred):
        """Восстановление b из a и slope"""
        if abs(a_pred) < 1e-10:
            return 0.01
        slope_rad = slope_pred * (math.pi / 180)
        return math.tan(slope_rad) / abs(a_pred)

    def _create_slope_scorer(self):
        """Нормализованный скорер для (f0, slope)"""

        def normalized_scorer(estimator, X, y):
            y_pred = estimator.predict(X)

            mae_f0 = np.mean(np.abs(y[:, 0] - y_pred[:, 0]))
            f0_mean = np.mean(y[:, 0])
            mae_f0_normalized = mae_f0 / f0_mean if f0_mean > 0 else mae_f0

            mae_slope = np.mean(np.abs(y[:, 1] - y_pred[:, 1]))
            mae_slope_normalized = mae_slope / 90.0

            w_slope = 5.0
            return -(mae_f0_normalized + w_slope * mae_slope_normalized)

        return make_scorer(normalized_scorer, greater_is_better=False)

    def _create_coeff_scorer(self):
        """Скорер напрямую оптимизирует FE и SE"""

        def fe_se_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            fe_errors = []
            se_errors = []

            for i in range(len(y)):
                f0 = X[i, 4]
                f0_pred = X[i, 7]
                slope_pred = X[i, 8]

                a_true, d_true = y[i]
                a_pred, d_pred = y_pred[i]

                b_true = self._compute_b_from_slope(a_true, slope_pred)
                b_pred = self._compute_b_from_slope(a_pred, slope_pred)

                fe = self._compute_frequency_error(
                    [a_true, b_true, f0_pred, d_true],
                    [a_pred, b_pred, f0_pred, d_pred],
                    f0
                )
                se = self._compute_slope_error(
                    [a_true, b_true, f0_pred, d_true],
                    [a_pred, b_pred, f0_pred, d_pred]
                )

                fe_errors.append(fe)
                se_errors.append(se)

            return -(np.mean(fe_errors) + np.mean(se_errors) * 100)

        return make_scorer(fe_se_scorer, greater_is_better=False)

    def train(self, test_size=0.2, n_iter_slope=30, n_iter_coeff=30, cv_splits=5):
        """
        Обучение двухстадийной модели.

        Параметры:
        ----------
        test_size : float
            Размер тестовой выборки
        n_iter_slope : int
            Число итераций RandomizedSearch для slope predictor
        n_iter_coeff : int
            Число итераций RandomizedSearch для coeff predictor
        cv_splits : int
            Число фолдов для кросс-валидации
        """
        # Первый сплит: отделяем тестовую выборку
        X_temp, X_test, y_temp, y_test, metadata_temp, metadata_test = train_test_split(
            self.x, self.y, self.metadata, test_size=test_size, random_state=self.random_state
        )

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"LINEAR COMBINED MODEL: обучение на {len(X_temp)} samples, тест на {len(X_test)} samples")
            print(f"{'=' * 80}\n")

        # ===== СТАДИЯ 1: Обучение slope predictor на ПОЛНОМ X_temp =====
        y_slope_full = self._prepare_slope_data(y_temp)
        self.slope_predictor = self._train_slope_predictor(X_temp, y_slope_full, n_iter_slope, cv_splits)

        # ===== Получение out-of-fold предсказаний для X_temp =====
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        slope_oof = cross_val_predict(
            self.slope_predictor, X_temp, y_slope_full,
            cv=kf, n_jobs=-1, verbose=0
        )

        if self.verbose:
            mae_f0 = np.mean(np.abs(y_slope_full[:, 0] - slope_oof[:, 0]))
            mae_slope = np.mean(np.abs(y_slope_full[:, 1] - slope_oof[:, 1]))
            print(f"Slope Predictor OOF MAE: f0={mae_f0:.4f} GHz, slope={mae_slope:.4f}°")

        # ===== СТАДИЯ 2: Обучение coeff predictor на ПОЛНОМ X_temp + OOF slope =====
        x_coeff_train, y_coeff_train = self._prepare_coeff_data(X_temp, slope_oof, y_temp)
        self.coeff_predictor = self._train_coeff_predictor(x_coeff_train, y_coeff_train, n_iter_coeff, cv_splits)

        # ===== Финальная оценка на тестовой выборке =====
        self._evaluate_combined_model(X_test, y_test, metadata_test)

        if self.verbose:
            print("\n" + "=" * 80)
            print("ОЦЕНКА ПЕРЕОБУЧЕНИЯ")
            print("=" * 80)
            print("\nSLOPE PREDICTOR:")
            slope_train_pred = self.slope_predictor.predict(X_temp)
            self._print_slope_metrics(y_slope_full, slope_train_pred, "Train")
            slope_test_pred = self.slope_predictor.predict(X_test)
            y_slope_test = self._prepare_slope_data(y_test)
            self._print_slope_metrics(y_slope_test, slope_test_pred, "Test")

    def _print_slope_metrics(self, y_true, y_pred, dataset_name):
        """Вывод метрик для slope predictor"""
        mae_f0 = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
        mae_slope = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
        print(f"  {dataset_name}: MAE f0={mae_f0:.4f} GHz, MAE slope={mae_slope:.4f}°")

    def _train_slope_predictor(self, x_train, y_slope_train, n_iter, cv_splits):
        """Обучение первой модели: X -> (f0, slope)"""
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        param_grid = {
            'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'ridge__fit_intercept': [True, False],
        }

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf,
            scoring=self._create_slope_scorer(),
            verbose=0, n_jobs=-1, random_state=self.random_state
        )

        search.fit(x_train, y_slope_train)

        if self.verbose:
            print(f"\nSLOPE MODEL (Linear):")
            print(f"  Лучшие параметры: {search.best_params_}")
            print(f"  Лучший score: {search.best_score_:.4f}")

        return search.best_estimator_

    def _train_coeff_predictor(self, x_coeff_train, y_coeff_train, n_iter, cv_splits):
        """Обучение второй модели: (X + slope) -> (a, d)"""
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        param_grid = {
            'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'ridge__fit_intercept': [True, False],
        }

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf,
            scoring=self._create_coeff_scorer(),
            verbose=0, n_jobs=-1, random_state=self.random_state
        )

        search.fit(x_coeff_train, y_coeff_train)

        if self.verbose:
            print(f"\nCOEFF MODEL (Linear):")
            print(f"  Лучшие параметры: {search.best_params_}")
            print(f"  Лучший score: {search.best_score_:.4f}\n")

        return search.best_estimator_

    def _evaluate_combined_model(self, x_test, y_test, metadata_test):
        """Оценка полной модели на тестовой выборке"""
        slope_predictions = self.slope_predictor.predict(x_test)
        x_coeff_test = np.hstack([x_test, slope_predictions])
        coeff_predictions = self.coeff_predictor.predict(x_coeff_test)

        full_predictions = []
        for i, (a_pred, d_pred) in enumerate(coeff_predictions):
            f0_pred, slope_pred = slope_predictions[i]
            b_pred = self._compute_b_from_slope(a_pred, slope_pred)
            c_pred = f0_pred

            constrained = self.apply_constraints([a_pred, b_pred, c_pred, d_pred], metadata_test[i][0])
            full_predictions.append(constrained)

        self._calculate_metrics(y_test, np.array(full_predictions), metadata_test)

    def _calculate_metrics(self, y_true, y_pred, metadata):
        """Вычисление всех метрик для финального отчёта"""
        mse_values = []
        mae_values = []
        me_values = []
        freq_errors = []
        slope_errors = []

        for i in range(len(y_true)):
            f0 = metadata[i][0]
            x_grid = np.linspace(f0 - 0.1 * f0, f0 + 0.1 * f0, 1000)

            true_curve = self.arctg_func(y_true[i], x_grid)
            pred_curve = self.arctg_func(y_pred[i], x_grid)

            mse_values.append(mean_squared_error(true_curve, pred_curve))
            mae_values.append(mean_absolute_error(true_curve, pred_curve))
            me_values.append(max_error(true_curve, pred_curve))

            freq_errors.append(self._compute_frequency_error(y_true[i], y_pred[i], f0))
            slope_errors.append(self._compute_slope_error(y_true[i], y_pred[i]))

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
            print(f"\n{'=' * 80}")
            print(f"ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
            print(f"{'=' * 80}")
            print(f"MSE: среднее={self.mean_mse:.4f}, макс={self.max_mse:.4f}")
            print(f"MAE: среднее={self.mean_mae:.4f}, макс={self.max_mae:.4f}")
            print(f"Max Error: среднее={self.mean_me:.4f}, макс={self.max_me:.4f}")
            print(f"FE: среднее={self.mean_freq_error:.4f}%, макс={self.max_freq_error:.4f}%")
            print(f"SE: среднее={self.mean_slope_error:.4f}, макс={self.max_slope_error:.4f}")
            print(f"{'=' * 80}\n")

    def predict(self, a_wg, b_wg, c_wg, d_wg, f0):
        """Предсказание для новых данных"""
        input_features = np.array([[a_wg, b_wg, c_wg, d_wg, f0, f0 * b_wg, a_wg * b_wg]])

        slope_pred = self.slope_predictor.predict(input_features)[0]
        f0_pred, slope_deg_pred = slope_pred

        extended_features = np.hstack([input_features, [[f0_pred, slope_deg_pred]]])
        a_pred, d_pred = self.coeff_predictor.predict(extended_features)[0]

        b_pred = self._compute_b_from_slope(a_pred, slope_deg_pred)
        c_pred = f0_pred

        return self.apply_constraints([a_pred, b_pred, c_pred, d_pred], f0)
