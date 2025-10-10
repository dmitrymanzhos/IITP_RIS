import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, make_scorer
import math
from base_predictor import BasePredictor


class RandomForestCombinedPredictor(BasePredictor):  # объединенная модель
    def __init__(self, verbose=True, show_plots=False, random_state=42):
        super().__init__(verbose, show_plots, random_state)
        self.slope_predictor = None
        self.coeff_predictor = None

    def _prepare_slope_data(self, y_data):  # подготовка входных признаков для первой модели
        y_slope = []
        for coeffs in y_data:
            f0, slope = self._compute_f0_slope(coeffs)
            y_slope.append([f0, slope])
        return np.array(y_slope)

    def _prepare_coeff_data(self, x_data, slope_predictions):  # подготовка входных признаков для второй модели
        x_coeff = np.hstack([x_data, slope_predictions])
        y_coeff = self.y[:, [0, 3]]
        return x_coeff, y_coeff

    def _compute_f0_slope(self, coeffs):
        a, b, c, d = coeffs
        f0 = c
        derivative = a * b
        angle_rad = math.atan(abs(derivative))
        slope = angle_rad * (180 / math.pi)
        return f0, slope

    def _compute_b_from_slope(self, a_pred, slope_pred):
        if abs(a_pred) < 1e-10:
            return 0.0
        slope_rad = slope_pred * (math.pi / 180)
        return math.tan(slope_rad) / abs(a_pred)

    def _create_slope_scorer(self):  # функция ошибок в виде SE
        def combined_mae_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            mae_f0 = np.mean(np.abs(y[:, 0] - y_pred[:, 0]))
            mae_slope = np.mean(np.abs(y[:, 1] - y_pred[:, 1]))
            return -(mae_f0 + mae_slope)

        return make_scorer(combined_mae_scorer, greater_is_better=False)

    def _create_coeff_scorer(self):  # функция ошибок в виде MAE
        def curve_mae_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            errors = []
            for i in range(y.shape[0]):
                f0 = X[i, 7]
                a_pred, d_pred = y_pred[i]
                b_pred = self._compute_b_from_slope(a_pred, X[i, 8])
                c_pred = f0
                x_grid = np.linspace(f0 - 0.1 * f0, f0 + 0.1 * f0, 100)
                true_curve = self.arctg_func([y[i, 0], self._compute_b_from_slope(y[i, 0], X[i, 8]), f0, y[i, 1]],
                                             x_grid)
                pred_curve = self.arctg_func([a_pred, b_pred, c_pred, d_pred], x_grid)
                errors.append(mean_absolute_error(true_curve, pred_curve))
            return np.mean(errors)

        return make_scorer(curve_mae_scorer, greater_is_better=False)

    def train(self, test_size=0.2, n_iter=50, cv_splits=5):  # обучение обеих моделей
        X_temp, X_test, y_temp, y_test, metadata_temp, metadata_test = train_test_split(
            self.x, self.y, self.metadata, test_size=test_size, random_state=self.random_state
        )

        X_slope, X_coeff, y_slope, y_coeff, metadata_slope, metadata_coeff = train_test_split(
            X_temp, y_temp, metadata_temp, test_size=0.5, random_state=self.random_state
        )

        y_slope_train = self._prepare_slope_data(y_slope)
        self.slope_predictor = self._train_slope_predictor(X_slope, y_slope_train, n_iter, cv_splits)

        slope_predictions = self.slope_predictor.predict(X_coeff)
        x_coeff_train, y_coeff_train = self._prepare_coeff_data(X_coeff, slope_predictions)
        self.coeff_predictor = self._train_coeff_predictor(x_coeff_train, y_coeff_train, n_iter, cv_splits)

        self._evaluate_combined_model(X_test, y_test, metadata_test)
        if self.verbose:
            self.slope_predictor.evaluate_overfitting(X_slope, y_slope_train, metadata_slope)
            self.coeff_predictor.evaluate_overfitting(x_coeff_train, y_coeff_train, metadata_coeff)

    def _train_slope_predictor(self, x_train, y_slope_train, n_iter, cv_splits):  # обучение первой модели
        base_model = RandomForestRegressor(random_state=self.random_state, bootstrap=False)
        model = MultiOutputRegressor(base_model)
        param_grid = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [2, 3],
            'estimator__min_samples_split': [5, 10, 15],
            'estimator__min_samples_leaf': [2, 3, 4],
            'estimator__max_features': ['sqrt', 'log2']
        }
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf, scoring=self._create_slope_scorer(),
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )
        search.fit(x_train, y_slope_train)
        if self.verbose:
            print(f"Slope модель - лучшие параметры: {search.best_params_}")
            print(f"Slope модель - лучший score: {search.best_score_:.4f}")
        return search.best_estimator_

    def _train_coeff_predictor(self, x_coeff_train, y_coeff_train, n_iter, cv_splits):  # обучение второй модели
        base_model = RandomForestRegressor(random_state=self.random_state, bootstrap=False)
        model = MultiOutputRegressor(base_model)
        param_grid = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [2, 3],  # ограничил глубину для избежания переобучения
            'estimator__min_samples_split': [5, 10, 15],
            'estimator__min_samples_leaf': [2, 3, 4],
            'estimator__max_features': ['sqrt', 'log2']
        }
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf, scoring=self._create_coeff_scorer(),
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )
        search.fit(x_coeff_train, y_coeff_train)
        if self.verbose:
            print(f"Coeff модель - лучшие параметры: {search.best_params_}")
            print(f"Coeff модель - лучший score: {search.best_score_:.4f}")
        return search.best_estimator_

    def _evaluate_combined_model(self, x_test, y_test, metadata_test):  # оценка работы модели
        slope_predictions = self.slope_predictor.predict(x_test)
        x_coeff_test = np.hstack([x_test, slope_predictions])
        coeff_predictions = self.coeff_predictor.predict(x_coeff_test)
        full_predictions = []
        for i, (a_pred, d_pred) in enumerate(coeff_predictions):
            f0_pred, slope_pred = slope_predictions[i]
            b_pred = self._compute_b_from_slope(a_pred, slope_pred)
            c_pred = f0_pred
            full_predictions.append([a_pred, b_pred, c_pred, d_pred])
        self._calculate_metrics(y_test, np.array(full_predictions), metadata_test)

    def _calculate_metrics(self, y_true, y_pred, metadata):  # вычисление метрик
        mse_values = []
        mae_values = []
        me_values = []
        freq_errors = []
        slope_errors = []

        for i in range(len(y_true)):
            f0 = metadata[i][0]
            x_grid = np.linspace(f0 - 0.1 * f0, f0 + 0.1 * f0, 1000)
            true_curve = self.arctg_func(y_true[i], x_grid)
            pred_curve = self.arctg_func(self.apply_constraints(y_pred[i], f0), x_grid)

            mse_values.append(mean_squared_error(true_curve, pred_curve))
            mae_values.append(mean_absolute_error(true_curve, pred_curve))
            me_values.append(max_error(true_curve, pred_curve))

            a_true, b_true, c_true, d_true = y_true[i]
            a_pred, b_pred, c_pred, d_pred = self.apply_constraints(y_pred[i], f0)

            try:
                x0_true = c_true + np.tan(-d_true / a_true) / b_true
            except:
                x0_true = c_true
            try:
                x0_pred = c_pred + np.tan(-d_pred / a_pred) / b_pred
            except:
                x0_pred = c_pred

            freq_errors.append((abs(x0_true - x0_pred) / f0) * 100)

            slope_true = a_true * b_true
            slope_pred = a_pred * b_pred
            if abs(slope_true) < 1e-10:
                slope_error = abs(slope_pred)
            else:
                slope_error = abs(slope_true - slope_pred) / abs(slope_true)
            slope_errors.append(slope_error)

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

    def predict(self, a_wg, b_wg, c_wg, d_wg, f0):
        input_features = [a_wg, b_wg, c_wg, d_wg, f0, f0 * b_wg, a_wg * b_wg]

        slope_pred = self.slope_predictor.predict([input_features])[0]
        f0_pred, slope_deg_pred = slope_pred

        extended_features = input_features + [f0_pred, slope_deg_pred]
        a_pred, d_pred = self.coeff_predictor.predict([extended_features])[0]

        b_pred = self._compute_b_from_slope(a_pred, slope_deg_pred)
        c_pred = f0_pred

        return self.apply_constraints([a_pred, b_pred, c_pred, d_pred], f0)

    # def apply_constraints(self, coeffs, f0=None):
    #     a, b, c, d = coeffs
    #     if a >= -80:
    #         a = -80.01
    #     if b <= 0:
    #         b = 0.01
    #     if f0 is not None:
    #         c = np.clip(c, f0 - 0.05 * f0, f0 + 0.05 * f0)
    #     return [a, b, c, d]
