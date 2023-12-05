import gestion_donnees as gd
import SVC as SVC
import KNN

def main():
    donnees = gd.GestionDonnees("./data/train.csv")
    datas=donnees.transformation("species")
    X_train,X_test,t_train,t_test = gd.separation(datas)
    # svc = SVC.SVC()
    # svc.recherche_parametres(X_train,t_train)
    # svc.afficher_parametres()
    # svc.entrainement(X_train,t_train)
    knn = KNN.KNN()
    knn.recherche_parametres(X_train,t_train)
    knn.afficher_parametres()
    knn.entrainement(X_train,t_train)

if __name__ == "__main__":
    main()
