import itertools
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.Superclases.superclases import Clasificador
from src.tools.metricas import Metricas

'''Este módulo contiene herramientas útiles para el manejo de modelos de clasificación.
'''


class Herramientas:
    '''Clase que contiene métodos útiles para el manejo de modelos de clasificación.
    '''
    @staticmethod
    def cross_validation(features: pd.DataFrame, target: pd.Series, classifier, k_fold: int = 5, metrica: str = "accuracy", promedio: str = "binario", verbose = False) -> float:
        '''Realiza validación cruzada de un clasificador.

        Args:
            features (pd.DataFrame): Conjunto de datos de entrenamiento.
            target (pd.Series): Etiquetas de los datos de entrenamiento.
            classifier (Clasificador): Clasificador a evaluar.
            k_fold (int): Cantidad de folds a utilizar.
            metrica (str): Métrica a utilizar para evaluar el clasificador. Puede ser 'accuracy' o 'f1'.
            promedio (str): Tipo de promedio a utilizar en la métrica 'f1'. Puede ser 'binario', 'micro', 'macro' o 'ponderado'.
            verbose (bool): Indica si se deben imprimir los scores de cada fold.

        Returns:
            float: Score promedio de la validación cruzada.
        '''
        if metrica == "accuracy":
            score = Metricas.accuracy_score
        elif metrica == "f1":
            score = lambda y_true, y_pred: Metricas.f1_score(y_true, y_pred, promedio)
        else:
            raise ValueError("Métrica no soportada")
        
        lista_indices = features.index.to_list()
        regist_grupos = len(lista_indices) // k_fold
        groups = []

        for i in range(k_fold):
            desde = i * regist_grupos
            if i == k_fold-1:
                hasta = len(lista_indices)
            else:
                hasta = desde + regist_grupos
            groups.append(lista_indices[desde:hasta])
        
        k_score_total = 0
        for j in range(k_fold):
            X_test = features.loc[groups[j]]
            y_test = target.loc[groups[j]]
            X_train = features.loc[~features.index.isin(groups[j])]
            y_train = target.loc[~target.index.isin(groups[j])]
            
            clasificador = classifier.__class__(**classifier.__dict__)
            clasificador.fit(X_train, y_train)
            predicciones = clasificador.predict(X_test)
            k_score = score(y_test, predicciones)
            
            k_score_total += k_score
            
            if verbose: print(f"Score individual del fold nro {j+1}:", k_score)
            
        return k_score_total / k_fold

    @staticmethod
    def dividir_set(X, y, test_size=0.2, val_size=0.2, val=False, random_state=None) -> tuple:
        '''Divide un conjunto de datos en entrenamiento, validación y prueba.

        Args:
            X (pd.DataFrame): Conjunto de datos de entrenamiento.
            y (pd.Series): Etiquetas de los datos de entrenamiento.
            test_size (float): Proporción de datos a utilizar como prueba.
            val_size (float): Proporción de datos a utilizar como validación.
            val (bool): Indica si se debe dividir en validación.
            random_state (int): Semilla para la generación de números aleatorios.

        Returns:
            tuple: Conjuntos de datos divididos.
        '''
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        X = X.iloc[indices]
        y = y.iloc[indices]
        
        n_total = len(X)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size) if val else 0
        
        X_test, y_test = X.iloc[:n_test], y.iloc[:n_test]
        X_val, y_val = X.iloc[n_test:n_test+n_val], y.iloc[n_test:n_test+n_val]
        X_train, y_train = X.iloc[n_test+n_val:], y.iloc[n_test+n_val:]
        
        if val:
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test

class GridSearch:
    '''Clase que realiza una búsqueda en grilla de hiperparámetros para un clasificador.
    '''
    def __init__(self, clasificador: Clasificador, params: dict[str, list], cv: bool = True, k_fold: int = 5, random_state: Optional[int] = None) -> None:
        '''Inicializa la clase GridSearch.

        Args:
            clasificador (Clasificador): Clasificador a utilizar.
            params (dict[str, list]): Diccionario con los hiperparámetros a probar.
            cv (bool): Indica si se debe realizar validación cruzada.
            k_fold (int): Cantidad de folds a utilizar en la validación cruzada.
            random_state (int): Semilla para la generación de números aleatorios.
        '''
        self._clasificador: Clasificador = clasificador
        self._params: dict[str, list] = params
        self._cv: bool = cv
        self._k_fold: int = k_fold
        self.mejores_params: Optional[dict] = None
        self.mejor_score: float = 0
        self.mejor_modelo: Optional[Clasificador] = None
        self.random_state = random_state
        self.resutados: dict = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''Realiza la búsqueda en grilla de hiperparámetros.

        Args:
            X (pd.DataFrame): Conjunto de datos de entrenamiento.
            y (pd.Series): Etiquetas de los datos de entrenamiento.
        '''
        params = list(self._params.keys())
        self.resutados = {p: [] for p in params}
        self.resutados['score'] = []
        valores = list(self._params.values())
        
        for combinacion in itertools.product(*valores):
            parametros = dict(zip(params, combinacion))
            
            print(f'Probando combinación: {parametros}')
            clasificador = self._clasificador.__class__()
            for p in parametros:
                setattr(clasificador, p, parametros[p])
                self.resutados[p].append(parametros[p])
            if self._cv:
                score = Herramientas.cross_validation(X, y, clasificador, self._k_fold)
                print(f'Score: {score}')
            else:
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state) # usariamos el nuestro, pero este es mas eficiente xq no usa pandas supongo
                clasificador.fit(x_train, y_train)
                score = Metricas.accuracy_score(y_val, clasificador.predict(x_val))
            if score > self.mejor_score:
                self.mejor_score = score
                self.mejores_params = parametros
            self.resutados['score'].append(score)

    def mostrar_resultados(self):
        '''Presenta los resultados de la búsqueda en grilla.

        Returns:
            pd.DataFrame: DataFrame con los resultados de la búsqueda en grilla.        
        '''
        return pd.DataFrame(self.resutados).sort_values(by='score', ascending=False)         