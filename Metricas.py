import pandas as pd
# TODO: investigar y agregar: recall_score, precision_score, roc_auc_score, log-loss, etc.  (no creo que sea necesario)

class Metricas:

    @staticmethod
    def f1_score(y_true: pd.Series, y_pred: list, promedio = "binario") -> float | dict:

        if len(y_true) != len(y_pred):
            raise ValueError("y_true e y_pred debe tener la misma longitud")
        
        y_true = y_true.tolist()

        # recopilo datos que sirven para todo tipo de promedio
        f1_scores = []
        soportes = []
        verdaderos_p_total = 0
        falsos_p_total = 0
        falsos_n_total = 0
        verdaderos_n_total = 0

        
        combinada = list(zip(y_true, y_pred))
        clases = set(y_true) if len(y_true) >= len(y_pred) else set(y_pred)

        for clase in clases:
            
            verdaderos_p = sum(1 for y_t, y_p in combinada if y_p == clase and y_t == clase)
            verdaderos_n = sum(1 for y_t, y_p in combinada if y_p != clase and y_t != clase)
            falsos_p = sum(1 for y_t, y_p in combinada if y_p == clase and y_t != clase)
            falsos_n = sum(1 for y_t, y_p in combinada if y_p != clase and y_t == clase)

            verdaderos_p_total += verdaderos_p
            verdaderos_n_total += verdaderos_n
            falsos_p_total += falsos_p
            falsos_n_total += falsos_n
            
            precision = verdaderos_p / (verdaderos_p + falsos_p) if (verdaderos_p + falsos_p) > 0 else 0
            recall = verdaderos_p / (verdaderos_p + falsos_n) if (verdaderos_p + falsos_n) > 0 else 0

            soporte = verdaderos_p + falsos_n
            soportes.append(soporte)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)

        # devuelvo el f1 score segun el promedio requerido
        
        if promedio == "binario":
            if len(clases) != 2:
                raise ValueError("Promedio binario no es válido en problemas multiclase")
            
            f1_score = f1_scores[0] # considero la primera clase que aparece como la positiva, podria ver como usar pos_label

        elif promedio == "macro":
            f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        elif promedio == "micro":
            precision_global = verdaderos_p_total / (verdaderos_p_total + falsos_p_total)
            recall_global = verdaderos_p_total / (verdaderos_p_total + falsos_n_total)

            f1_score = 2 * (precision_global * recall_global) / (precision_global + recall_global)

        elif promedio == "ponderado":
            soporte_total = sum(soportes)

            f1_score = 0
            for i, f1 in enumerate(f1_scores):
                f1_score += f1*(soportes[i]/soporte_total)
        
        elif promedio is None: # devuelve el f1_score de cada clase
            retorno = {}
            for i, clase in enumerate(clases):
                retorno[clase] = f1_scores[i]

            return retorno

        return f1_score

    @staticmethod
    def accuracy_score(y_true: pd.Series, y_pred: list):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true e y_pred debe tener la misma longitud")
        
        combinada = list(zip(y_true, y_pred))
        verdaderos_p = sum(1 for y_t, y_p in combinada if y_t == y_p)
        return verdaderos_p / len(y_true)
    
    @staticmethod
    def error(y_true: pd.Series, y_pred: list):
        return 1 - Metricas.accuracy_score(y_true, y_pred)

    