import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from .base_predictor import BasePredictor


class RandomForestPredictor(BasePredictor):
    def __init__(self, verbose=True, show_plots=False, random_state=42):
        super().__init__(verbose, show_plots, random_state)

    def train(self, test_size=0.2, n_iter=50, cv_splits=5):
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            self.x, self.y, self.metadata, test_size=test_size, random_state=self.random_state
        )

        if self.verbose:
            print(f"Обучение на {len(X_train)} samples, тест на {len(X_test)} samples")

        model = RandomForestRegressor(random_state=self.random_state)

        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 4, 5, 6, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9, None],
        }

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf,
            scoring='neg_mean_absolute_error',  # self._create_fe_se_scorer(),
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        if self.verbose:
            print(f"Лучшие параметры: {search.best_params_}")
            print(f"Лучший score (-(FE + SE*100)): {search.best_score_:.4f}")
            self.evaluate_overfitting(X_train, y_train, metadata_train)

        y_pred_test = self.model.predict(X_test)
        self._evaluate_curves(X_test, y_test, y_pred_test, metadata_test)
