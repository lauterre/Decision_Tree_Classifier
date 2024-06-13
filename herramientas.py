import itertools
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from _superclases import Clasificador
from metricas import Metricas


class Herramientas:
    @staticmethod
    def cross_validation(features: pd.DataFrame, target: pd.Series, classifier, k_fold: int = 5, metrica: str = "accuracy", promedio: str = "binario", verbose = False) -> float:
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
    def __init__(self, clasificador: Clasificador, params: dict[str, list], cv: bool = True, k_fold: int = 5, random_state: Optional[int] = None):
        self._clasificador: Clasificador = clasificador
        self._params: dict[str, list] = params
        self._cv: bool = cv
        self._k_fold: int = k_fold
        self.mejores_params: Optional[dict] = None
        self.mejor_score: float = 0
        self.mejor_modelo: Optional[Clasificador] = None
        self.random_state = random_state
        self.resutados: dict = {}
    
    def fit(self, X, y):
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
                x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
                clasificador.fit(x_train, y_train)
                score = Metricas.accuracy_score(y_val, clasificador.predict(x_val))
            if score > self.mejor_score:
                self.mejor_score = score
                self.mejores_params = parametros
            self.resutados['score'].append(score)


    def mostrar_resultados(self):
        return pd.DataFrame(self.resutados).sort_values(by='score', ascending=False)            
            
