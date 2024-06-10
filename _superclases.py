from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Optional
import pandas as pd
from _impureza import Impureza
from graficador import GraficadorArbol
from metricas import Metricas

class Clasificador(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
    
class Hiperparametros:
    def __init__(self, **kwargs):
        self.max_prof: int = kwargs.get('max_prof', -1)
        self.min_obs_nodo: int = kwargs.get('min_obs_nodo', -1)
        self.min_infor_gain: float = kwargs.get('min_infor_gain', -1.0) # me hace ruido que este aca, esta atado a la entropia CART no lo usaria
        self.min_obs_hoja: int = kwargs.get('min_obs_hoja', -1)

class Arbol(ABC): # seria ArbolNario
    def __init__(self) -> None:
        self.subs: list[Arbol]= []
    
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
    
    def posorden(self) -> list["Arbol"]:
        recorrido = []
        for subarbol in self.subs:
            recorrido += subarbol.posorden()
        recorrido.append(self)
        return recorrido
    
    @abstractmethod
    def agregar_subarbol(self, subarbol):
        raise NotImplementedError
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError
    
    @abstractmethod
    def es_raiz(self):
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
        self.signo_split_anterior: Optional[str] = None
        self.clase: Optional[str] = None
        self.impureza: Optional[Impureza] = None
    
    def es_raiz(self) -> bool:
        return self.valor_split_anterior is None
    
    def _values(self) -> list[int]:
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
    
    def _total_samples(self) -> int:
        return len(self.data)
    
    def agregar_subarbol(self, subarbol) -> None:
        for key, value in self.__dict__.items():
            if key in Hiperparametros().__dict__:  # Solo copiar los atributos que están en Hiperparametros
                setattr(subarbol, key, value)
        self.subs.append(subarbol)
    
    def graficar(self) -> None:
        graficador = GraficadorArbol(self)
        graficador.graficar()

    def reduced_error_pruning(self, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        def _interna_rep(arbol: ArbolClasificador, x_test, y_test):
            if not arbol.es_hoja():
                for subarbol in arbol.subs:
                    _interna_rep(subarbol, x_test, y_test)

                pred_raiz: list[str] = arbol.predict(x_test)
                error_clasif_raiz = Metricas.error(y_test, pred_raiz)

                error_clasif_ramas = 0.0

                for subarbol in arbol.subs:
                    pred_podada = subarbol.predict(x_test) # type: ignore
                    error_clasif_podada = Metricas.error(y_test, pred_podada)
                    error_clasif_ramas = error_clasif_ramas + error_clasif_podada

                if error_clasif_ramas < error_clasif_raiz:
                    print(" * Podar \n")
                    arbol.subs = []
                else:
                    print(" * No podar \n")

        _interna_rep(self, x_test, y_test)

    
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, ArbolClasificador) and (self.data.equals(__value.data)
                                                           and self.target.equals(__value.target) 
                                                           and self.atributo_split == __value.atributo_split 
                                                           and self.atributo_split_anterior == __value.atributo_split_anterior 
                                                           and self.valor_split_anterior == __value.valor_split_anterior 
                                                           and self.signo_split_anterior == __value.signo_split_anterior 
                                                           and self.clase == __value.clase)

    def _podar(self, nodo: "ArbolClasificador") -> "ArbolClasificador":
        arbol_podado = deepcopy(self)
        def _interna(arbol, nodo):
            if arbol == nodo:
                arbol.subs = []
            else:
                for sub in arbol_podado.subs:
                    sub._podar(nodo)
        _interna(arbol_podado, nodo)
        return arbol_podado
    
    def reduced_error_pruning2(self, x_val, y_val, margen: float = 0) -> "ArbolClasificador":
        # TODO: check si esta entrenado
        arbol_completo = deepcopy(self)
        error_inicial = Metricas.error(y_val, arbol_completo.predict(x_val))
        nodos = arbol_completo.posorden()
        for nodo in nodos:
            if not nodo.es_hoja():
                arbol_podado = arbol_completo._podar(nodo)
                nuevo_error = Metricas.error(y_val, arbol_podado.predict(x_val))
                if nuevo_error - margen < error_inicial:
                    arbol_completo = arbol_podado
                    error_inicial = nuevo_error
        return arbol_completo

    @abstractmethod
    def _mejor_atributo_split(self) -> str | None:
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str) -> None:
        raise NotImplementedError

class Bosque(ABC):
    def __init__(self, cantidad_arboles: int = 10) -> None:
        self.arboles: list[Arbol] = []
        self.cantidad_arboles = cantidad_arboles