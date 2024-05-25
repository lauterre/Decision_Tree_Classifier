from copy import deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Metricas import Metricas
from _superclases import ClasificadorBosque, Bosque, Hiperparametros, Arbol
from ArbolDecisionID3 import ArbolDecisionID3
from Graficador import TreePlot

class RandomForest(Bosque, ClasificadorBosque):
    def __init__(self,**kwargs):
        super().__init__()
        ClasificadorBosque.__init__(self, **kwargs)

    def seleccionar_atributos(self, X):
        n_features = X.shape[1]
        if self.cantidad_atributos == 'sqrt':
            size = int(np.sqrt(n_features))
        elif self.cantidad_atributos == 'log2':
            size = int(np.log2(n_features))
        else:
            size = n_features

        indices = np.random.choice(n_features, size, replace=False)
        return indices


    def fit(self, X, y):
        for _ in range(self.cantidad_arboles):
            # Bootstrapping
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Selección de atributos
            atributos = self.seleccionar_atributos(X_sample)
            X_sample = X_sample[:, atributos]

            # Crear y entrenar un nuevo árbol
            if self.clase_arbol == 'id3':
                arbol = ArbolDecisionID3(max_prof=self.max_prof, min_obs_nodo=self.min_obs_nodo)
                arbol.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                self.arboles.append(arbol)

    def predict(self, X):
        # Lista para almacenar las predicciones de cada árbol
        todas_predicciones = pd.DataFrame(index=X.index, columns=range(len(self.arboles)))  # Ahora es DataFrame
        
        for i, arbol in enumerate(self.arboles):
            todas_predicciones[i] = arbol.predict(X)  # Guardar las predicciones en el DataFrame

        # Aplicar la votación mayoritaria
        predicciones_finales = todas_predicciones.apply(lambda x: x.value_counts().idxmax(), axis=1)
        
        return predicciones_finales
    

if __name__ == "__main__":
    # Crea un conjunto de datos de ejemplo
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target

    # Entrena el RandomForest con ArbolDecisionID3
    rf = RandomForest(clase_arbol=ArbolDecisionID3, cantidad_arboles=10, cantidad_atributos='sqrt', max_prof=10, mon_obs_nodo=2)
    rf.fit(X, y)

    # Predice con el RandomForest
    predicciones = rf.predict(pd.DataFrame(X))
    print(predicciones)

        