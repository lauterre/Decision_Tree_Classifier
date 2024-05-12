from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd

class Clasificador(ABC):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1):
        self.max_prof = max_prof
        self.min_obs_nodo = min_obs_nodo

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
    
class ArbolDecision(ABC):
    def __init__(self) -> None:
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series]= None
        self.atributo: Optional[str] = None
        self.categoria: Optional[str]= None
        self.target_categorias: Optional[list[str]]= None
        self.clase: Optional[str] = None
        self.subs: list[ArbolDecision]= []

    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abstractmethod
    def _mejor_split(self):
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str, valor: Any) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def entropia(self):
        raise NotImplementedError
    
    @abstractmethod
    def _information_gain(self, atributo: str) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def es_raiz(self):
        raise NotImplementedError
    
    @abstractmethod
    def es_hoja(self):
        raise NotImplementedError
    
    @abstractmethod
    def altura(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def imprimir(self) -> None:
        raise NotImplementedError