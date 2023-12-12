import gestion_donnees as gd
import SVC
import KNN
import reseau_de_neurones
import modele_generatif
import logistic
import evaluation_metrics
import GBT
import perceptron
import time


def main():
    donnees = gd.GestionDonnees("./data/train.csv")
    datas = donnees.transformation("species")
    X_train, X_test, t_train, t_test = donnees.separation(datas)

    generatif = modele_generatif.ModeleGeneratif()
    knn = KNN.KNN()
    tree = GBT.GBT()
    perceptron_ = perceptron.PerceptronModel()
    logistic_ = logistic.Logistic()
    svc = SVC.SVC()
    reseau_neurones = reseau_de_neurones.ReseauDeNeurones()

    methodes = [generatif, knn, tree, perceptron_, logistic_, svc, reseau_neurones]

    ev = evaluation_metrics.evaluation_metrics(t_test)

    for methode in methodes:
        debut = time.time()
        methode.recherche_parametres(X_train, t_train)
        methode.afficher_parametres()
        methode.entrainement(X_train, t_train)
        pred = methode.prediction(X_test)
        ev.evaluate(methode, pred)
        ev.confusion_mat(pred)
        delta = int(time.time() - debut)
        print(f"Temps écoulé : {delta // 60} min {delta % 60} sec\n")


if __name__ == "__main__":
    main()
