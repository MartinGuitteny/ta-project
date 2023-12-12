import gestion_donnees as gd
import evaluation_metrics as evm

import SVC
import KNN
import reseau_de_neurones
import modele_generatif
import logistic

import GBT
import perceptron

import argparse

import numpy as np
import time

MODELS = [
    "generative",
    "knn",
    "gbtree",
    "perceptron",
    "logistic",
    "svc",
    "neural_network"
]

# def main():
#     donnees = gd.GestionDonnees("./data/train.csv")
#     datas = donnees.transformation("species")
#     X_train, X_test, t_train, t_test = donnees.separation(datas)

#     generatif = modele_generatif.ModeleGeneratif()
#     knn = KNN.KNN()
#     tree = GBT.GBT()
#     perceptron_ = perceptron.PerceptronModel()
#     logistic_ = logistic.Logistic()
#     svc = SVC.SVC()
#     reseau_neurones = reseau_de_neurones.ReseauDeNeurones()

#     methodes = [generatif, knn, tree, perceptron_, logistic_, svc, reseau_neurones]

#     ev = evm.evaluation_metrics(t_test)

#     for methode in methodes:
#         debut = time.time()
#         methode.recherche_parametres(X_train, t_train)
#         methode.afficher_parametres()
#         methode.entrainement(X_train, t_train)
#         pred = methode.prediction(X_test)
#         ev.evaluate(methode, pred)
#         ev.confusion_mat(pred,methode.name())
#         delta = int(time.time() - debut)
#         print(f"Temps écoulé : {delta // 60} min {delta % 60} sec\n")

def do_n_simulations(N, model_type, verbose):
    
    precision = []
    recall = []
    f1_score = []
    duration = []

    for i in range(N):
        
        if verbose >= 1:
            print(f"Iteration {i + 1} / {N}")

        data_gestion = gd.GestionDonnees("./data/train.csv")
        data = data_gestion.transformation("species")
        X_train, X_test, t_train, t_test = data_gestion.separation(data)
        model = None

        if model_type == "generative":
            model = modele_generatif.ModeleGeneratif()
        elif model_type == "knn":
            model = KNN.KNN()
        elif model_type == "gbtree":
            model = GBT.GBT()
        elif model_type == "perceptron":
            model = perceptron.PerceptronModel()
        elif model_type == "logistic":
            model= logistic.Logistic()
        elif model_type == "svc":
            model = SVC.SVC()
        else:
            model = reseau_de_neurones.ReseauDeNeurones()

        ev = evm.EvaluationMetrics(t_test)

        beginning = time.time()

        model.recherche_parametres(X_train, t_train)
        if verbose >= 2:
            model.afficher_parametres()
        model.entrainement(X_train, t_train)
        prediction = model.prediction(X_test)
        prec, rec, f1 = ev.evaluate(model, prediction, verbose)
        delta = int(time.time() - beginning)

        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)
        duration.append(delta)

        if verbose >= 1:
            print(f"Time : {delta // 60} min {delta % 60} sec\n")

    print(f'''Precision : 
Mean = {round(np.mean(precision), 3)}
Std = {round(np.std(precision), 3)}

Recall : 
Mean = {round(np.mean(recall), 3)}
Std = {round(np.std(recall), 3)}

F1-Score : 
Mean = {round(np.mean(f1_score), 3)}
Std = {round(np.std(f1_score), 3)}

Mean time : {int(np.mean(duration) // 60)} min {int(np.mean(duration) % 60)} sec''')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("-m", "--model", type=str, default="neural_network", help="Model")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbose level")
    args = parser.parse_args()

    if args.number_iterations > 0 and args.model in MODELS:
        do_n_simulations(args.number_iterations, args.model, args.verbose)
    else:
        print(f"Wrong input. Number of iterations must be > 0. Model must be in {MODELS}.")
