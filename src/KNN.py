from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self,n=5):
        self.n=n
        self.model = KNeighborsClassifier(n_neighbors=self.n)

    def entrainement(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def recherche_parametres(self, X, t, cv=10):
        param_grid = {'n_neighbors': [5,7,10,12,15]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, t)
        self.C = grid_search.best_params_.get("C")
        self.kernel = grid_search.best_params_.get("kernel")
        self.gamma = grid_search.best_params_.get("gamma")
        self.model = grid_search.best_estimator_
    
    def afficher_parametres(self):
        print(f'''Parametres:
n :{self.n}''')
        
    def name(self):
        return "K-nearest neighbors"