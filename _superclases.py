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
    
class Hiperparametros:
    def __init__(self, **kwargs):
        self.max_prof = kwargs.get('max_prof', -1)
        self.min_obs_nodo = kwargs.get('min_obs_nodo', -1)
        self.min_infor_gain = kwargs.get('min_infor_gain', -1.0)
        self.min_obs_hoja = kwargs.get('min_obs_hoja', -1)

class ClasificadorArbol(Clasificador, ABC):
    def __init__(self, **kwargs):
        hiperparametros = Hiperparametros(**kwargs)        
        for key, value in hiperparametros.__dict__.items():
            setattr(self, key, value)

class Arbol(ABC):
    def __init__(self) -> None:
        self.data: pd.DataFrame 
        self.target: pd.Series
        self.atributo_split: Optional[str] = None
        self.atributo_split_anterior: Optional[str] = None
        self.valor_split_anterior: Optional[str]= None
        self.target_categorias: Optional[list[str]]= None
        self.clase: Optional[str] = None
        self.subs: list[Arbol]= []
    
    def es_raiz(self):
        return self.valor_split_anterior is None
    
    def es_hoja(self):
        return self.subs == []
    
    def __len__(self) -> int:
        if self.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.subs])
        
    def _values(self):
        recuento_values = self.target.value_counts()
        values = []
        for valor in self.target_categorias:
            value = recuento_values.get(valor, 0)
            values.append(value)
        return values
    
    def altura(self) -> int:
        altura_actual = 0
        for subarbol in self.subs:
            altura_actual = max(altura_actual, subarbol.altura())
        return altura_actual + 1

    def _total_samples(self):
        return len(self.data)
    
    @abstractmethod
    def agregar_subarbol(self, subarbol):
        raise NotImplementedError
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def _mejor_atributo_split(self):
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str, valor: Any = None) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _entropia(self):
        raise NotImplementedError       #Este método no se si va aca, creo que solo es ID3. C4.5 usa la Ganancia de Informacion normalizada
    
    @abstractmethod
    def _information_gain(self, atributo: str, valor:Any = None) -> float:
        raise NotImplementedError    
    
    @abstractmethod
    def imprimir(self) -> None:
        raise NotImplementedError
    