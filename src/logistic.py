from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class Logistic:
    def __init__(self, C=1):
        self.C = C
        self.model = LogisticRegression(C=C)

    def entrainement(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def recherche_parametres(self, X, y, cv=5):
        param_grid = {'C': [0.01,0.1,1,10]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, y)
        self.C = grid_search.best_params_.get("C")
        self.model = grid_search.best_estimator_

    def afficher_parametres(self):
        print(f'''Paramètres:
C : {self.C}''')

    def name(self):
        return "Régression logistique"
