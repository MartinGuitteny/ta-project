from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV


class PerceptronModel:
    def __init__(self, max_iter=1000, eta0=0.1):
        self.max_iter = max_iter
        self.eta0 = eta0
        self.model = Perceptron(max_iter=self.max_iter, eta0=self.eta0)

    def entrainement(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def recherche_parametres(self, X, y, cv=10):
        param_grid = {"max_iter": [10000], "eta0": [1, 0.1, 0.01, 0.001]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, y)
        self.max_iter = grid_search.best_params_.get("max_iter")
        self.eta0 = grid_search.best_params_.get("eta0")
        self.model = grid_search.best_estimator_

    def afficher_parametres(self):
        print(
            f"""Parameters:
max_iter: {self.max_iter}
eta0: {self.eta0}"""
        )

    def name(self):
        return "Perceptron"
