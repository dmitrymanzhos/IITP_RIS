import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from .base_predictor import BasePredictor


class LinearPredictor(BasePredictor):
    def __init__(self, verbose=True, show_plots=False, random_state=42):
        super().__init__(verbose, show_plots, random_state)

    def train(self, test_size=0.2, n_iter=30, cv_splits=5):
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            self.x, self.y, self.metadata, test_size=test_size, random_state=self.random_state
        )

        if self.verbose:
            print(f"Обучение на {len(X_train)} samples, тест на {len(X_test)} samples")

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

        param_grid = {
            'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'ridge__fit_intercept': [True, False],
        }

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf,
            scoring='neg_mean_absolute_error',
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        if self.verbose:
            print(f"Лучшие параметры: {search.best_params_}")
            print(f"Лучший score: {search.best_score_:.4f}")

        # Сначала вычисляем метрики на test
        y_pred_test = self.model.predict(X_test)
        self._evaluate_curves(X_test, y_test, y_pred_test, metadata_test)
