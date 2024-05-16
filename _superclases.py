from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd

class Clasificador(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

class ClasificadorArbol(Clasificador, ABC):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1):
        self.max_prof = max_prof
        self.min_obs_nodo = min_obs_nodo
    
class Arbol(ABC):
    def __init__(self) -> None:
        self.data: pd.DataFrame 
        self.target: pd.Series
        self.atributo: Optional[str] = None
        self.categoria: Optional[str]= None
        self.target_categorias: Optional[list[str]]= None
        self.clase: Optional[str] = None
        self.subs: list[Arbol]= []
    
    def es_raiz(self):
        return self.categoria is None
    
    def es_hoja(self):
        return self.subs == []
    
    def __len__(self) -> int:
        if self.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.subs])
    
    @abstractmethod
    def _mejor_split(self):
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str, valor: Any) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def entropia(self):
        raise NotImplementedError       #Este método no se si va aca, creo que solo es ID3. C4.5 usa la Ganancia de Informacion normalizada
    
    @abstractmethod
    def _information_gain(self, atributo: str) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def altura(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def imprimir(self) -> None:
        raise NotImplementedError