import pandas as pd
# TODO: investigar y agregar: recall_score, precision_score, roc_auc_score, log-loss, etc.
from typing import Union, Dict
from src.Excepciones.excepciones import LongitudInvalidaException, PromedioInvalidoException

class Metricas:
    '''Clase que contiene métodos para calcular métricas de evaluación de clasificadores.
    '''
    @staticmethod
    def f1_score(y_true: pd.Series, y_pred: list, promedio = "binario") -> float | dict:
        '''Calcula el F1 Score de un clasificador.

        Args:
            y_true (pd.Series): Valores del target reales.
            y_pred (list): Valores del target predichos.
            promedio (str): Tipo de promedio a utilizar. Puede ser 'binario', 'micro', 'macro', 'ponderado' o None.

        Returns:
            float | dict: F1 Score o diccionario con el F1 Score de cada clase.
        '''

        if len(y_true) != len(y_pred):
            raise LongitudInvalidaException("Error: Longitud de y_true y y_pred no coinciden")
        
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
                raise PromedioInvalidoException("Promedio binario no es válido en problemas multiclase")
            
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
        
        elif promedio is None:
            retorno = {}
            for i, clase in enumerate(clases):
                retorno[clase] = f1_scores[i]

            return retorno

        return f1_score

    @staticmethod
    def accuracy_score(y_true: pd.Series, y_pred: list) -> float:
        '''Calcula el Accuracy de un clasificador.

        Args:
            y_true (pd.Series): Valores del target reales.
            y_pred (list): Valores del target predichos.

        Returns:
            float: Accuracy Score.
        '''
        if len(y_true) != len(y_pred):
            raise LongitudInvalidaException("Error: Longitud de y_true y y_pred no coinciden")
        
        combinada = list(zip(y_true, y_pred))
        verdaderos_p = sum(1 for y_t, y_p in combinada if y_t == y_p)
        return verdaderos_p / len(y_true)
    
    @staticmethod
    def error_score(y_true: pd.Series, y_pred: list) -> float:
        '''Calcula el Error de un clasificador.

        Args:
            y_true (pd.Series): Valores del target reales.
            y_pred (list): Valores del target predichos.

        Returns:
            float: Error Score.
        '''
        return 1 - Metricas.accuracy_score(y_true, y_pred)

    