from abc import abstractmethod
from typing import Callable
import numpy as np
import pandas as pd

'''Este módulo contiene la clase abstracta Impureza y sus subclases Entropia, Gini y MSE.

La clase abstracta Impureza contiene los métodos abstractos calcular y calcular_impureza_split.
La subclase Entropia implementa los métodos calcular y calcular_impureza_split.
La subclase Gini implementa el método calcular.
La subclase MSE implementa el método calcular.
'''

class Impureza():
    '''Clase abstracta que define los métodos necesarios para calcular la impureza de un target y la ganancia de información de un atributo.
    '''
    @abstractmethod
    def calcular(self, target: pd.Series) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def calcular_impureza_split(self, arbol, atributo: str, split: Callable) -> float:
        raise NotImplementedError
    
class Entropia(Impureza):
    def calcular(self, target: pd.Series) -> float:
        '''Esta funcion calcula la entropia de un target dado.

        Args:
            target (pd.Series): target a calcular la entropia.

        Returns:
            float: entropia del target        
        '''
        entropia = 0
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia -= proporcion * np.log2(proporcion)
        return entropia
 
    def calcular_impureza_split(self, arbol, atributo: str, split: Callable): 
        '''Esta funcion calcula la ganancia de información para un atributo dado.   

        Args:
            arbol (Arbol): arbol a calcular la impureza.
            atributo (str): atributo a splittear.
            split (Callable): funcion de split.

        Returns:
            float: ganancia de información del atributo.        
        '''
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

### Para CART, seguro no los usamos
    
class Gini(Impureza):
    '''Clase que implementa el cálculo de la impureza Gini.
    '''
    def calcular(self, target: pd.Series) -> float:
        '''Esta funcion calcula la impureza Gini de un target dado.
        
        Args:
            target (pd.Series): target a calcular la impureza Gini.

        Returns:    
            float: impureza Gini del target.
        '''
        gini = 1
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            gini -= proporcion ** 2
        return gini
    
    def calcular_impureza_split(self, atributo:str, target: pd.Series) -> float: # funcion de costo
        ...
    
    def __str__(self) -> str:
        return "Gini"
    
class MSE(Impureza):
    '''Clase que implementa el cálculo de la impureza  partir del error cuadrático medio.
    '''
    def calcular(self, target: pd.Series) -> float:
        '''Esta funcion calcula el error cuadrático medio de un target dado.
        
        Args:
            target (pd.Series): target a calcular el error cuadrático medio.

        Returns:    
            float: error cuadrático medio del target.
        '''
        media = target.mean()
        mse = ((target - media) ** 2).mean()
        return mse
    
    def __str__(self) -> str:
        return "MSE"