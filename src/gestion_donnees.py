from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


class GestionDonnees:
    def __init__(self, path_csv):
        self.path = path_csv

    def transformation(self, attribut):
        df = pd.read_csv(self.path)
        le = LabelEncoder().fit(df[attribut])
        df[attribut] = le.transform(df[attribut])
        return df

    def separation(self, df, size=0.2):
        # t = df["species"]
        # x = df.drop(columns=["species", "id"])
        X_train, X_test, t_train, t_test = self.balanced_train_test_split(df, test_size=size)
        return X_train, X_test, t_train, t_test

    def balanced_train_test_split(self, df, test_size=0.2):
        """
        Sépare un data frame en 4 : X_train, X_test, t_train, t_test.
        Pour tout data frame parmi ces 4, il y a autant d'individus de chaque classe.
        Exemple avec size=0.2 et 100 classes de 10 individus :
        8 individus de chaque classe dans X_train et t_train
        2 individus de chaque classe dans X_test et t_test
        """
        df = pd.DataFrame(df.drop(columns="id").reset_index(drop=True).sample(frac=1))
        number_tests_samples = 10 * test_size # 10 samples per class

        selected_tests_samples_per_class = {}
        train_index = []
        test_index = []

        for class_ in pd.unique(df["species"]):
            selected_tests_samples_per_class.update({class_: 0})

        for row in df.itertuples():
            if selected_tests_samples_per_class[row.species] < number_tests_samples:
                test_index.append(row.Index)
                selected_tests_samples_per_class[row.species] += 1
            else:
                train_index.append(row.Index)
        
        # print(train_index, len(train_index))
        # print(test_index, len(test_index))

        X_train = df.loc[train_index]
        X_test = df.loc[test_index]
        
        t_train = X_train["species"]
        t_test = X_test["species"]

        X_train = X_train.drop(columns="species")
        X_test = X_test.drop(columns="species")

        # print(X_train.head(5), X_test.head(5), t_train.head(5), t_test.head(5))

        return X_train, X_test, t_train, t_test
            


