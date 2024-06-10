from abc import abstractmethod
from typing import Callable
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

    
    def _information_gain_base(self, arbol, atributo: str, split: Callable):
        entropia_actual = self.calcular(arbol.target)
        len_actual = arbol._total_samples()
        nuevo = arbol.copy()

        split(nuevo, atributo)

        entropias_subarboles = 0 
        for subarbol in nuevo.subs:
            entropia = self.calcular(subarbol.target)
            len_subarbol = subarbol._total_samples()
            entropias_subarboles += ((len_subarbol/len_actual) * entropia)

        information_gain = entropia_actual - entropias_subarboles
        return information_gain
    
    def __str__(self) -> str:
        return "Entropia"

class Gini(Impureza):
    def calcular(self, target: pd.Series) -> float:
        gini = 1
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            gini -= proporcion ** 2
        return gini
    
    def __str__(self) -> str:
        return "Gini"
    