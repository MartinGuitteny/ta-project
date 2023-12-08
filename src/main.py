import gestion_donnees as gd
import SVC
import KNN
import reseau_de_neurones
import modele_generatif
import logistic
import evaluation_metrics

def main():
    donnees = gd.GestionDonnees("./data/train.csv")
    datas=donnees.transformation("species")
    X_train,X_test,t_train,t_test = gd.separation(datas)

    generatif = modele_generatif.ModeleGeneratif()
    svc = SVC.SVC()
    knn = KNN.KNN()
    reseau_neurones = reseau_de_neurones.ReseauDeNeurones()
    logistic_ = logistic.Logistic()

    methodes = [svc, knn, reseau_neurones, generatif, logistic_]
    ev = evaluation_metrics.evaluation_metrics(t_test)

    for methode in methodes:
        methode.recherche_parametres(X_train,t_train)
        methode.afficher_parametres()
        methode.entrainement(X_train,t_train)
        pred = methode.prediction(X_test)
        ev.evaluate(methode,pred)

if __name__ == "__main__":
    main()
