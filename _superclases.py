from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional
import pandas as pd
import numpy as np

class Clasificador(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
    
class Hiperparametros:
    def __init__(self, **kwargs):
        # No sería mas claro dejarlos en None si no estan definidos?
        self.max_prof: int = kwargs.get('max_prof', -1)
        self.min_obs_nodo: int = kwargs.get('min_obs_nodo', -1)
        self.min_infor_gain: float = kwargs.get('min_infor_gain', -1.0)
        self.min_obs_hoja: int = kwargs.get('min_obs_hoja', -1)
        self.criterio: str = kwargs.get('criterio', 'entropia')

class Arbol(ABC): # seria ArbolNario
    def __init__(self) -> None:
        # moví los atributos que habia acá porque considero que no son propios de un arbol, sino de un modelo
        self.subs: list[Arbol]= []
    
    @abstractmethod
    def es_raiz(self):
        raise NotImplementedError
    
    def es_hoja(self):
        return self.subs == []
        
    def __len__(self) -> int: # cantidad de nodos
        if self.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.subs])
    
    def altura(self) -> int:
        altura_actual = 0
        for subarbol in self.subs:
            altura_actual = max(altura_actual, subarbol.altura())
        return altura_actual + 1
    
    def ancho(self, nivel=0, nodos=defaultdict(int)):
        nodos[nivel] += 1
        for subarbol in self.subs:
            subarbol.ancho(nivel + 1, nodos)
        return max(nodos.values())
    
    def ancho_nivel(self, nivel=0, nodos=defaultdict(int)):
        nodos[nivel] += 1
        for subarbol in self.subs:
            subarbol.ancho_nivel(nivel + 1, nodos)
        return nodos[nivel]
    
    @abstractmethod
    def agregar_subarbol(self, subarbol):
        raise NotImplementedError
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError
    
    # TODO: pasar a __str__()
    @abstractmethod
    def imprimir(self) -> None:
        raise NotImplementedError
    
class ArbolClasificador(Arbol, Clasificador, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        hiperparametros = Hiperparametros(**kwargs)  
        for key, value in hiperparametros.__dict__.items():
            setattr(self, key, value)
        self.data: pd.DataFrame
        self.target: pd.Series
        self.target_categorias: Optional[list[str]]= None
        self.atributo_split: Optional[str] = None
        self.atributo_split_anterior: Optional[str] = None
        self.valor_split_anterior: Optional[str]= None
        self.clase: Optional[str] = None
    
    def es_raiz(self):
        return self.valor_split_anterior is None
    
    def _values(self):
        recuento_values = self.target.value_counts()
        values = []
        for valor in self.target_categorias:
            value = recuento_values.get(valor, 0)
            values.append(value)
        return values
    
    def set_clase(self) -> None:
        self.clase = self.target.value_counts().idxmax()

    def set_target_categorias(self, y) -> None:
        self.target_categorias = y.unique()
    
    def _total_samples(self):
        return len(self.data)
    
    def _puede_splitearse(self, prof_acum) -> bool:
        return not (len(self.target.unique()) == 1 or len(self.data.columns) == 0
                    or (self.max_prof != -1 and self.max_prof <= prof_acum)
                    or (self.min_obs_nodo != -1 and self.min_obs_nodo > self._total_samples()))
    
    def _entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize=True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia if entropia != 0 else 0
    
    def agregar_subarbol(self, subarbol):
        for key, value in self.__dict__.items():
            if key in Hiperparametros().__dict__:  # Solo copiar los atributos que están en Hiperparametros
                setattr(subarbol, key, value)
        self.subs.append(subarbol)

    @abstractmethod
    def _mejor_atributo_split(self) -> str | None:
        raise NotImplementedError
    
    @abstractmethod
    def _information_gain(self, atributo: str) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str) -> None:
        raise NotImplementedError
    

# Esto no lo toque porque aun no lo miré
    
# Habria que seguir la misma logica del ArbolClasificador
# BosqueClasificador deberia ser el RandomForest
# quizas la clase bosque no va, por ahora agrega solo dos comportamientos tipicos de un bosque: self.arboles, self.cantidad_arboles


class BosqueClasificador(Clasificador, ABC):
    def __init__(self,**kwargs) -> None:
        hiperparametros_arbol = Hiperparametros(**kwargs)
        for key, value in hiperparametros_arbol.__dict__.items():
            setattr(self, key, value)

class Bosque(ABC):
    def __init__(self, clase_arbol: str = "id3", cantidad_arboles: int = 10, cantidad_atributos:str ='sqrt') -> None:
        self.arboles: list[Arbol] = []
        self.cantidad_arboles = cantidad_arboles
        # estos atributos son propios de un modelo, deberian estar en BosqueClasificador
        self.cantidad_atributos = cantidad_atributos
        self.clase_arbol = clase_arbol

    # estos metodos tambien
    @staticmethod
    def _bootstrap_samples(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        n_samples = X.shape[0]
        atributos = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[atributos], y.iloc[atributos]
    
    @abstractmethod
    def seleccionar_atributos(self, X: pd.DataFrame)-> list[int]:
        raise NotImplementedError