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
import numpy as np


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
        ev.confusion_mat(pred,methode.name())
        delta = int(time.time() - debut)
        print(f"Temps écoulé : {delta // 60} min {delta % 60} sec\n")

def test_n_fois(n,methode):
    precision = []
    recall = []
    f1_score = []
    temps = []
    for i in range(n):
        donnees = gd.GestionDonnees("./data/train.csv")
        datas = donnees.transformation("species")
        X_train, X_test, t_train, t_test = donnees.separation(datas)

        if methode == "generatif":
            modele= modele_generatif.ModeleGeneratif()
        if methode == "knn":
            modele = KNN.KNN()
        if methode == "tree":
            modele = GBT.GBT()
        if methode == "perceptron":
            modele = perceptron.PerceptronModel()
        if methode == "logistic":
            modele= logistic.Logistic()
        if methode == "svc":
            modele = SVC.SVC()
        if methode == "reseau_neurones":
            modele = reseau_de_neurones.ReseauDeNeurones()
        ev = evaluation_metrics.evaluation_metrics(t_test)

        debut = time.time()
        modele.recherche_parametres(X_train, t_train)
        modele.afficher_parametres()
        modele.entrainement(X_train, t_train)
        pred = modele.prediction(X_test)
        prec,rec,f1 = ev.evaluate(modele, pred)
        delta = int(time.time() - debut)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)
        temps.append(delta)
    print(f'''precision : 
moyenne = {np.mean(precision)}
ecart-type = {np.std(precision)}

recall : 
moyenne = {np.mean(recall)}
ecart-type = {np.std(recall)}

f1_score : 
moyenne = {np.mean(f1_score)}
ecart-type = {np.std(f1_score)}

temps moyen : {np.mean(temps)}''')
if __name__ == "__main__":
    test_n_fois(5,"knn")
