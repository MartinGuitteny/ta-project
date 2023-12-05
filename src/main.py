import gestion_donnees as gd
import SVC as SVC
import KNN
from sklearn.metrics import recall_score, precision_score


def main():
    donnees = gd.GestionDonnees("./data/train.csv")
    datas=donnees.transformation("species")
    X_train,X_test,t_train,t_test = gd.separation(datas)
    svc = SVC.SVC()
    knn = KNN.KNN()
    methodes = [svc,knn]
    for methode in methodes:
        methode.recherche_parametres(X_train,t_train)
        methode.afficher_parametres()
        methode.entrainement(X_train,t_train)
        pred = methode.prediction(X_test)
        precision = precision_score(t_test,pred,average='macro')
        recall = recall_score(t_test,pred,average='macro')
        print(f'''Parametres:
modele :{methode.name()}
precision : {precision}
recall : {recall}''')

if __name__ == "__main__":
    main()
