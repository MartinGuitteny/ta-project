from sklearn.metrics import recall_score, precision_score

class evaluation_metrics():
    def __init__(self,t_test):
        self.t_test = t_test

    def evaluate(self,methode,pred):
        precision = precision_score(self.t_test,pred,average='macro', zero_division = 0)
        recall = recall_score(self.t_test,pred,average='macro', zero_division = 0)
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f'''Modèle : {methode.name()}
Précision : {precision}
Rappel : {recall}
F1-Score : {f1_score}''')
        return precision, recall, f1_score