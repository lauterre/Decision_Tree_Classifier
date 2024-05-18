from ArbolDecisionID3 import ArbolDecisionID3
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from _superclases import ClasificadorArbol, Arbol
from typing import Any, Optional
#------------------------------------------------------------------------------
# * Se implementa el método _split_continuo para manejar la división de 
# atributos continuos.
# * Se implementa el cálculo del Gain Ratio en el método 
# _information_gain_ratio_continuo.
# * Se modificó el método fit para que divida los atributos continuos 
# si es necesario.
# Se agregaron los métodos podar y _weighted_data para el manejo 
# de podado por reglas (a implementar)
#-------------------------------------------------------------------------------
class ArbolDecisionC45(ArbolDecisionID3):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1) -> None:
        super().__init__(max_prof, min_obs_nodo)
    
    # función que busca el mejor atributo para dividir el conjunto de datos,
    def _split_continuo(self, atributo: str) -> None:
        # Ordenar los valores únicos en orden ascendente
        valores_ordenados = sorted(self.data[atributo].unique())
        # Seleccionar puntos medios como posibles umbrales
        umbrales = [(valores_ordenados[i] + valores_ordenados[i+1]) / 2 for i in range(len(valores_ordenados) - 1)]
        # Calcular la ganancia de información para cada umbral
        mejor_ig_ratio = -1
        mejor_umbral = None
        for umbral in umbrales:
            ig_ratio = self._information_gain_ratio_continuo(atributo, umbral)
            if ig_ratio > mejor_ig_ratio:
                mejor_ig_ratio = ig_ratio
                mejor_umbral = umbral
        # Dividir el atributo en base al mejor umbral
        self._split(atributo, mejor_umbral)

    # Implementación del cálculo del Gain Ratio para atributos continuos 
    def _information_gain_ratio_continuo(self, atributo: str, umbral: float) -> float:
        # Divide el conjunto de datos en dos grupos basados en el umbral
        grupo_izquierdo = self.data[self.data[atributo] <= umbral]
        grupo_derecho = self.data[self.data[atributo] > umbral]
        # Calcula la entropía de los grupos resultantes
        entropia_grupo_izquierdo = self._entropy(grupo_izquierdo)
        entropia_grupo_derecho = self._entropy(grupo_derecho)
        # Calcula la entropía del conjunto de datos respecto al atributo
        entropia_atributo = (len(grupo_izquierdo) / len(self.data)) * entropia_grupo_izquierdo \
                            + (len(grupo_derecho) / len(self.data)) * entropia_grupo_derecho
        # Calcula la entropía del conjunto de datos respecto a la distribución del atributo
        entropia_distribucion_atributo = self._entropy(self.data[atributo])
        # Calcula el split info
        split_info = self._entropy(grupo_izquierdo[atributo]) + self._entropy(grupo_derecho[atributo])
        # Calcula el gain ratio
        if split_info != 0:
            gain_ratio = (entropia_distribucion_atributo - entropia_atributo) / split_info
        else:
            gain_ratio = 0
        return gain_ratio

    def _entropy(self, data: pd.Series) -> float:
        proporciones = data.value_counts(normalize=True)
        entropia = -(proporciones * np.log2(proporciones)).sum()
        return entropia if not np.isnan(entropia) else 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target = y
        self.data = X
        self.clase = self.target.value_counts().idxmax()
        def _interna(arbol: ArbolDecisionC45, prof_acum: int = 0):
            arbol.target_categorias = y.unique()
            if prof_acum == 0:
                prof_acum = 1
            if not ( len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                    or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                    or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples() ) ):
                mejor_atributo = arbol._mejor_split()
                if isinstance(arbol.data[mejor_atributo].dtype, (float, int)):
                    arbol._split_continuo(mejor_atributo)
                else:
                    arbol._split(mejor_atributo)
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum+1)
        _interna(self)

# A implemtar --------------------------------------------------   
    # Implementa el proceso de podado por reglas
    def podar(self):
        pass
    # Implementa el manejo de datos ponderados
    def _weighted_data(self, X: pd.DataFrame, y: pd.Series):
        pass
#----------------------------------------------------------------------------  
    def accuracy_score(self, y_true: pd.Series, y_pred: list[str]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("Las longitudes de y_true y y_pred deben ser iguales.")
        correctas = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        precision = correctas / len(y_true)
        return precision

def probar_C45(df, target:str):
        X = df.drop(target, axis=1)
        y = df[target]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        arbol = ArbolDecisionC45(min_obs_nodo=20)
        arbol.fit(x_train, y_train)
        arbol.imprimir()
        y_pred = arbol.predict(x_test)
        accuracy = arbol.accuracy_score(y_test, y_pred)
        print(f"\naccuracy: {accuracy}")
        print(f"cantidad de nodos: {len(arbol)}")
        print(f"altura: {arbol.altura()}\n")    
#----------------------------------------------------------------------------
        
if __name__ == "__main__":
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    probar_C45(iris_df, "target")


  