from abc import abstractmethod
import numpy as np
import pandas as pd

class Impureza():
    @abstractmethod
    def calcular(self, target: pd.Series) -> float:
        raise NotImplementedError
    
class Entropia(Impureza):
    def calcular(self, target: pd.Series) -> float:
        entropia = 0
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia -= proporcion * np.log2(proporcion)
        return entropia

class Gini(Impureza):
    def calcular(self, target: pd.Series) -> float:
        gini = 1
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            gini -= proporcion ** 2
        return gini
    
class ErrorClasificacion(Impureza):
    def calcular(self, target: pd.Series) -> float:
        proporciones = target.value_counts(normalize=True)
        max_proporcion = proporciones.max()
        return 1 - max_proporcion
    