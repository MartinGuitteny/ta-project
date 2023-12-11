from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

    def confusion_mat(self,t_pred):
        classes = []
        ConfusionMatrixDisplay.from_predictions(self.t_test,t_pred,include_values=False, display_labels=classes)
        plt.show()
