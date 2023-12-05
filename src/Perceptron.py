from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter, learning_rate_init=self.learning_rate_init)

    def entrainement(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def recherche_parametres(self, X, y, cv=10):
        param_grid = {'hidden_layer_sizes': [(50, 50), (100, 100)],
                      'max_iter': [1000],
                      'learning_rate_init': [0.001, 0.01]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, y)
        self.hidden_layer_sizes = grid_search.best_params_.get("hidden_layer_sizes")
        self.max_iter = grid_search.best_params_.get("max_iter")
        self.learning_rate_init = grid_search.best_params_.get("learning_rate_init")
        self.model = grid_search.best_estimator_

    def afficher_parametres(self):
        print(f'''Parameters:
hidden_layer_sizes: {self.hidden_layer_sizes}
max_iter: {self.max_iter}
learning_rate_init: {self.learning_rate_init}''')

    def name(self):
        return "Perceptron multi-couches"
