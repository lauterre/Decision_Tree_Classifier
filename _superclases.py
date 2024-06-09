from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Optional
import pandas as pd
import _impureza
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
        self.min_infor_gain: float = kwargs.get('min_infor_gain', -1.0)
        self.min_obs_hoja: int = kwargs.get('min_obs_hoja', -1)
        self.criterio_impureza: str = kwargs.get('criterio_impureza', 'Entropia')
        criterios_posibles = {name: cls for name, cls in vars(_impureza).items() if isinstance(cls, type)}
        try:
            Impureza = getattr(_impureza, self.criterio_impureza)
        except AttributeError:
            raise ValueError(f"Criterio de impureza no válido, criterios válidos: {list(criterios_posibles.keys())[1:]}")
        self.impureza = Impureza()

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
    
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, ArbolClasificador) and self.__dict__ == __value.__dict__
    
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
    
    def _puede_splitearse(self, prof_acum: int, mejor_atributo: str) -> bool:
        copia = self.copy() # OJO
        information_gain = self._information_gain(mejor_atributo)
        # hacer el split hipotetico y ver si se supera min_obs_hoja
        copia._split(mejor_atributo)
        for subarbol in copia.subs:
            if self.min_obs_hoja != -1 and subarbol._total_samples() < self.min_obs_hoja:
                return False
            
        return not (len(self.target.unique()) == 1 or len(self.data.columns) == 0
                    or (self.max_prof != -1 and self.max_prof <= prof_acum)
                    or (self.min_obs_nodo != -1 and self.min_obs_nodo > self._total_samples())
                    or (self.min_infor_gain != -1 and self.min_infor_gain > information_gain))
    
    def agregar_subarbol(self, subarbol):
        for key, value in self.__dict__.items():
            if key in Hiperparametros().__dict__:  # Solo copiar los atributos que están en Hiperparametros
                setattr(subarbol, key, value)
        self.subs.append(subarbol)
    
    def _impureza(self):
        return self.impureza.calcular(self.target)
    
    def graficar(self):
        plotter = GraficadorArbol(self)
        plotter.plot()

    def podar(self, nodo: "ArbolClasificador") -> "ArbolClasificador":
        arbol_podado = deepcopy(self)
        def _interna(arbol, nodo):
            if arbol == nodo:
                arbol.subs = []
            else:
                for sub in arbol_podado.subs:
                    sub.podar(nodo)
        _interna(arbol_podado, nodo)
        return arbol_podado
    
    def reduced_error_pruning2(self, x_val, y_val) -> "ArbolClasificador":
        # TODO: check si esta entrenado
        arbol_completo = deepcopy(self)
        error_inicial = Metricas.error(y_val, arbol_completo.predict(x_val))
        nodos = arbol_completo.posorden()
        for nodo in nodos:
            if not nodo.es_hoja():
                arbol_podado = arbol_completo.podar(nodo)
                nuevo_error = Metricas.error(y_val, arbol_podado.predict(x_val))
                if nuevo_error < error_inicial:
                    arbol_completo = arbol_podado
                    error_inicial = nuevo_error
        return arbol_completo


    @abstractmethod
    def _mejor_atributo_split(self) -> str | None:
        raise NotImplementedError
    
    @abstractmethod
    def _information_gain(self, atributo: str) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str) -> None:
        raise NotImplementedError

class Bosque(ABC):
    def __init__(self, cantidad_arboles: int = 10) -> None:
        self.arboles: list[Arbol] = []
        self.cantidad_arboles = cantidad_arboles