from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture

class ModeleGeneratif:
    def __init__(self, n_components=99, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)

    def entrainement(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def recherche_parametres(self, X, y, cv=5):
        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical']}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, y)
        self.covariance_type = grid_search.best_params_.get("covariance_type")
        self.model = grid_search.best_estimator_

    def afficher_parametres(self):
        print(f'''Paramètres :
covariance_type : {self.covariance_type}''')

    def name(self):
        return "Modèle génératif"