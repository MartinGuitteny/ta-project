from sklearn.model_selection import GridSearchCV
from sklearn import svm

class SVC:
    def __init__(self,C=1,gam=0.1,ker='rbf'):
        self.C = C
        self.kernel = ker
        self.gamma = gam

        self.model = svm.SVC(C = self.C,kernel = self.kernel,gamma = self.gamma)

    def entrainement(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def prediction(self, X_test):
        return self.model.predict(X_test)

    def recherche_parametres(self, X, t, cv=10):
        param_grid = {'C': [0.001,0.01,0.1, 1,10], 'kernel': ['linear', 'rbf','sigmoid','poly'], 'gamma': [0.001,0.01,0.1, 1]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, t)
        self.C = grid_search.best_params_.get("C")
        self.kernel = grid_search.best_params_.get("kernel")
        self.gamma = grid_search.best_params_.get("gamma")
        self.model = grid_search.best_estimator_
    
    def afficher_parametres(self):
        print(f'''Parametres:
C :{self.C}
kernel : {self.kernel}
gamma : {self.gamma}''')
    
    def name(self):
        return "Support Vector Classification"