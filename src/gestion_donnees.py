from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

class GestionDonnees:
    def __init__(self,path_csv):
        self.path = path_csv

    def transformation(self,attribut):
        df = pd.read_csv(self.path)
        le = LabelEncoder().fit(df[attribut])
        df[attribut] = le.transform(df[attribut])
        return df
    
def separation(df,size=0.2):
    t=df["species"]
    x= df.drop(columns=['species','id'])
    X_train, X_test, t_train, t_test = train_test_split(x, t, test_size=size)
    return X_train,X_test,t_train,t_test
    