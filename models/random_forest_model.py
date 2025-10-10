import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from base_predictor import BasePredictor


class RandomForestPredictor(BasePredictor):  # обычная модель random forest
    def __init__(self, verbose=True, show_plots=False, n_estimators=100,
                 max_depth=None, random_state=42):
        super().__init__(verbose, show_plots, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

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
        base_model = RandomForestRegressor(
            random_state=self.random_state,
            bootstrap=False
        )
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
            model, param_grid, n_iter=n_iter, cv=kf, scoring=self._create_scorer(),
            verbose=self.verbose, n_jobs=-1, random_state=self.random_state
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        if self.verbose:
            print(f"Лучшие параметры: {search.best_params_}")
            print(f"Лучший score: {search.best_score_:.4f}")
            self.evaluate_overfitting(X_train, y_train, metadata_train)
        y_pred_test = self.model.predict(X_test)
        self._evaluate_curves(X_test, y_test, y_pred_test, metadata_test)
        # if self.show_plots:
            # self._plot_feature_importance()

    # def _plot_feature_importance(self):
    #     """Визуализация важности признаков"""
    #     if hasattr(self.model.estimators_[0], 'feature_importances_'):
    #         # Усредняем важность признаков по всем выходным переменным
    #         importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
    #
    #         feature_names = ['a_wg', 'b_wg', 'c_wg', 'd_wg', 'f0', 'f0*b_wg', 'a_wg*b_wg']
    #
    #         plt.figure(figsize=(10, 6))
    #         indices = np.argsort(importances)[::-1]
    #
    #         plt.title("Важность признаков (Random Forest)")
    #         plt.bar(range(len(importances)), importances[indices])
    #         plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    #         plt.tight_layout()
    #
    #         os.makedirs('model_plots', exist_ok=True)
    #         plt.savefig('model_plots/random_forest_feature_importance.png')
    #
    #         if self.show_plots:
    #             plt.show()
    #         plt.close()