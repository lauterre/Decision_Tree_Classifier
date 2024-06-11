import itertools
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from _superclases import Clasificador
from metricas import Metricas

class Herramientas:
    @staticmethod
    def cross_validation(features: pd.DataFrame, target: pd.Series, classifier: Clasificador, k_fold:int, metrica: str = "accuracy", promedio: str = "binario") -> float:
        if metrica == "accuracy":
            score = Metricas.accuracy_score
        elif metrica == "f1":
            score = lambda y_true, y_pred: Metricas.f1_score(y_true, y_pred, promedio)
        else:
            raise ValueError("Métrica no soportada")
            
        
        lista_indices = features.index
        regist_grupos = len(lista_indices) // k_fold
        groups = []
        
        for i in range (k_fold):
            desde = i * regist_grupos
            if i == k_fold-1:
                hasta = len(lista_indices)
            else:
                hasta = desde + regist_grupos
                
            groups.append(lista_indices[desde:hasta])
        
        groups_X_train_CV = []
        groups_Y_train_CV = []
        groups_X_test_CV = []
        groups_Y_test_CV = []
        
        for j in range (k_fold):
            groups_X_test_CV.append ( features.loc[groups[j]] )
            groups_Y_test_CV.append ( target.loc[groups[j]] )
            _tempX = pd.DataFrame()
            _tempY = pd.DataFrame()
            for k in range (k_fold):
                if k != j:
                    _tempX = features.loc[groups[k]] if _tempX.empty else pd.concat ([_tempX, features.loc[groups[k]]] )    
                    _tempY = target.loc[groups[k]]   if _tempY.empty else pd.concat ([_tempY, target.loc[groups[k]]] )    

            groups_X_train_CV.append(_tempX)
            groups_Y_train_CV.append(_tempY)     
        
        k_score_total = 0
        
        for x in range (k_fold) :
            classifier.fit(groups_X_train_CV[x], groups_Y_train_CV[x])
            predicciones = classifier.predict(groups_X_test_CV[x])
            k_score = score(groups_Y_test_CV[x], predicciones)
            
            k_score_total += k_score
            print ("Score individual:", k_score)
            
        return k_score_total/k_fold

    @staticmethod
    def dividir_set(X, y, test_size=0.2, val_size=0.2, val=False, random_state=None):
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
    def __init__(self, clasificador: Clasificador, params: dict[str, list], cv: int = 5):
        self._clasificador: Clasificador = clasificador
        self._params: dict[str, list] = params
        self._cv: int = cv
        self.mejores_params: Optional[dict] = None
        self.mejor_score: float = 0
        self.mejor_modelo: Optional[Clasificador] = None
    
    def fit(self, X, y):
        params = list(self._params.keys())
        valores = list(self._params.values())

        for combinacion in itertools.product(*valores):
            parametros = dict(zip(params, combinacion))
            clasificador = self._clasificador.__class__()
            # clasificador = self._clasificador.__class__(**parametros)
            for p in parametros:
                setattr(clasificador, p, parametros[p])
            # solo para proabarlo
            x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            clasificador.fit(x_train, y_train)
            score = Metricas.accuracy_score(y_val, clasificador.predict(x_val))
            #score = Herramientas.cross_validation(X, y, clasificador, self._cv) # no funciona
            if score > self.mejor_score:
                self.mejor_score = score
                self.mejores_params = parametros
                self.mejor_modelo = clasificador
