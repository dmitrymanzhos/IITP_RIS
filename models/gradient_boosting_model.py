import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from base_predictor import BasePredictor


class GradientBoostingPredictor(BasePredictor):
    """
    Gradient Boosting модель с НАТИВНЫМ multi-output.

    КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
    - Убран MultiOutputRegressor — используется нативный GB multi-output
    - Добавлен early stopping через validation_fraction
    - Расширен диапазон n_estimators (до 500)
    - Скорректирован learning_rate (от 0.01)
    - Убран max_features=None (оставлены только sqrt и log2)
    - Используется FE/SE скорер
    """

    def __init__(self, verbose=True, show_plots=False, random_state=42):
        super().__init__(verbose, show_plots, random_state)

    def train(self, test_size=0.2, n_iter=50, cv_splits=5):
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            self.x, self.y, self.metadata, test_size=test_size, random_state=self.random_state
        )

        if self.verbose:
            print(f"Обучение на {len(X_train)} samples, тест на {len(X_test)} samples")

        # НАТИВНЫЙ GradientBoostingRegressor с early stopping
        model = GradientBoostingRegressor(
            random_state=self.random_state,
            validation_fraction=0.1,  # 10% для валидации
            n_iter_no_change=20,  # остановка, если 20 итераций без улучшения
            tol=1e-4
        )

        param_grid = {
            'n_estimators': [100, 200, 300, 500],  # Расширен верхний предел
            'learning_rate': [0.01, 0.05, 0.1, 0.15],  # Добавлен 0.01 для больших n_estimators
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2'],  # Убран None — избегаем полной корреляции
        }

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf,
            scoring=self._create_fe_se_scorer(),
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        if self.verbose:
            print(f"Лучшие параметры: {search.best_params_}")
            print(f"Лучший score (-(FE + SE*100)): {search.best_score_:.4f}")

            # Проверяем, сработал ли early stopping
            if hasattr(self.model, 'n_estimators_'):
                print(f"Фактическое количество деревьев (с early stopping): {self.model.n_estimators_}")

            self.evaluate_overfitting(X_train, y_train, metadata_train)

        y_pred_test = self.model.predict(X_test)
        self._evaluate_curves(X_test, y_test, y_pred_test, metadata_test)
