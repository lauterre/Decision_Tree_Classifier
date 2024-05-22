import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

class Metricas:

    def _precision(self, y_true, y_pred, clase):
        combinada = list(zip(y_true, y_pred))

        verdaderos_p = sum(1 for y_t, y_p in combinada if y_p == clase and y_t == clase)
        falsos_p = sum(1 for y_t, y_p in combinada if y_p == clase and y_t != clase)
        
        return verdaderos_p / (verdaderos_p + falsos_p) if (verdaderos_p + falsos_p) > 0 else 0

    def _recall(self, y_true, y_pred, clase):
        combinada = list(zip(y_true, y_pred))

        verdaderos_p = sum(1 for y_t, y_p in combinada if y_p == clase and y_t == clase)
        falsos_n = sum(1 for y_t, y_p in combinada if y_p != clase and y_t == clase)
        
        return verdaderos_p / (verdaderos_p + falsos_n) if (verdaderos_p + falsos_n) > 0 else 0

    @staticmethod
    def f1_score(y_true: pd.Series | list, y_pred: list, promedio = "binario"):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true e y_pred debe tener la misma longitud")
        
        if not isinstance(y_true, list):
            y_true = y_true.tolist()

        if promedio == "macro":
            f1_scores = []
            clases = pd.Series(y_true).unique()
            for clase in clases:
                # esto seria _recall y _precision pero un metodo estatico no tiene acceso
                combinada = list(zip(y_true, y_pred))
                verdaderos_p = combinada.count((clase, clase))
                falsos_p = sum(1 for y_t, y_p in combinada if y_p == clase and y_t != clase)
                falsos_n = sum(1 for y_t, y_p in combinada if y_p != clase and y_t == clase)
                
                precision = verdaderos_p / (verdaderos_p + falsos_p) if (verdaderos_p + falsos_p) > 0 else 0
                recall = verdaderos_p / (verdaderos_p + falsos_n) if (verdaderos_p + falsos_n) > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                f1_scores.append(f1)
            
            retorno = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        elif promedio == "binario":
            pass
        elif promedio == "micro":
            pass
        elif promedio == "ponderado": # seria weighted en sklearn
            pass
        elif promedio == "muestras":
            pass

        return retorno

    @staticmethod
    def accuracy_score(y_true: pd.Series | list, y_pred: list):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true e y_pred debe tener la misma longitud")
        
        combinada = list(zip(y_true, y_pred))
        verdaderos_p = sum(1 for y_t, y_p in combinada if y_t == y_p)
        return verdaderos_p / len(y_true)

if __name__ == "__main__":
    help(recall_score)