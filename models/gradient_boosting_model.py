import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from base_predictor import BasePredictor


class GradientBoostingPredictor(BasePredictor):
    def __init__(self, verbose=True, show_plots=False, random_state=42):
        super().__init__(verbose, show_plots, random_state)

    def _create_scorer(self):
        def curve_mae_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            errors = []
            for i in range(y.shape[0]):
                f0 = X[i, 4]
                x_grid = np.linspace(f0 - 0.1 * f0, f0 + 0.1 * f0, 100)
                true_curve = self.arctg_func(y[i], x_grid)
                pred_curve = self.arctg_func(self.apply_constraints(y_pred[i], f0), x_grid)
                errors.append(mean_absolute_error(true_curve, pred_curve))
            return np.mean(errors)

        return make_scorer(curve_mae_scorer, greater_is_better=False)

    def train(self, test_size=0.2, n_iter=50, cv_splits=5):
        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            self.x, self.y, self.metadata, test_size=test_size, random_state=self.random_state
        )

        if self.verbose:
            print(f"Обучение на {len(X_train)} samples, тест на {len(X_test)} samples")

        base_model = GradientBoostingRegressor(random_state=self.random_state)
        model = MultiOutputRegressor(base_model)

        param_grid = {
            'estimator__n_estimators': [50, 100, 200],
            'estimator__learning_rate': [0.05, 0.1, 0.15, 0.2],
            'estimator__max_depth': [3, 4, 5, 6],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__subsample': [0.8, 0.9, 1.0],
            'estimator__max_features': ['sqrt', 'log2', None]
        }
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=kf, scoring=self._create_scorer(),
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        if self.verbose:
            print(f"Лучшие параметры: {search.best_params_}")
            print(f"Лучший score: {search.best_score_:.4f}")

        y_pred_test = self.model.predict(X_test)
        self._evaluate_curves(X_test, y_test, y_pred_test, metadata_test)
