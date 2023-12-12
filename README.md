
# Projet IFT712

- Matéo DEMANGEON (demm1412)
- Martin GUITTENY (guim1106)
- Lucas RIOUX (riol2003)

## Utilisation

1. Accédez au répertoire du projet.

2. Installez les dépendances requises en utilisant pip dans votre environement virtuel.

    ```bash
    pip install -r requirements.txt
    ```

3. Depuis la racine du projet, exécutez le fichier principal.
    ```bash
    python3 src/main.py -n <nombre_d_iterations> -m <type_de_modele> -v <niveau_de_verbosité>
    ```

- `nombre_d_iterations` : Le nombre d'itérations pour les simulations.
- `type_de_modele` : Le type de modèle de classification à utiliser. Choisissez parmi : `generative`, `knn`, `gbtree`, `perceptron`, `logistic`, `svc`, `neural_network` .
- `niveau_de_verbosité` : Le niveau de verbosité pour la sortie, `0`, `1` ou `2`.
