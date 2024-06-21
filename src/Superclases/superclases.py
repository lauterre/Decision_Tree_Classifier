from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Optional
import pandas as pd
from src.Excepciones.excepciones import HiperparametroInvalidoException, ArbolNoEntrenadoException
from src.Impureza.impureza import Impureza
from src.tools.graficador import GraficadorArbol, GraficadorArbolFeo
from src.tools.metricas import Metricas

'''Este módulo contiene las superclases Clasificador, Arbol, ArbolClasificador y Bosque.
    - Clasificador es una clase abstracta que define los métodos necesarios para un modelo clasificador.
    - Arbol es una clase abstracta que define los métodos necesarios para la estructura del árbol.
    - ArbolClasificador es una clase abstracta que define los métodos necesarios para un árbol de decisión clasificador.
    - Bosque es una clase abstracta que define los métodos necesarios para un bosque de árboles.
'''

class Clasificador(ABC):
    '''Clase abstracta que define los métodos necesarios para un clasificador.'''
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
    
class Hiperparametros:
    '''Clase que contiene los hiperparámetros de un árbol de decisión.'''
    PARAMS_PERMITIDOS = {'max_prof', 'min_obs_nodo', 'min_infor_gain', 'min_obs_hoja'}
    def __init__(self, **kwargs):
        for key in kwargs:
            if key not in self.PARAMS_PERMITIDOS:
                raise HiperparametroInvalidoException(key)
        self.max_prof: int = kwargs.get('max_prof', -1)
        self.min_obs_nodo: int = kwargs.get('min_obs_nodo', -1)
        self.min_infor_gain: float = kwargs.get('min_infor_gain', -1.0)
        self.min_obs_hoja: int = kwargs.get('min_obs_hoja', -1)

class Arbol(ABC): 
    '''Clase abstracta que define los métodos necesarios para un árbol de decisión.
    Para este proyecto definimos un árbol de decisión como un árbol n-ario.
    '''
    def __init__(self) -> None:
        self.subs: list[Arbol]= []
    
    def es_hoja(self):
        '''Esta función devuelve True si el nodo es una hoja.
        
        Returns:    
            bool: True si el nodo es una hoja, False en caso contrario.'''
        return self.subs == []
        
    def __len__(self) -> int: 
        '''Esta función devuelve la cantidad de nodos del árbol.

        Returns:
            int: cantidad de nodos del árbol.

        '''
        if self.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.subs])
    
    def altura(self) -> int:
        '''Esta función devuelve la altura del árbol. Es decir los niveles del árbol.
        
        Returns:
            int: altura del árbol.'''
        altura_actual = 0
        for subarbol in self.subs:
            altura_actual = max(altura_actual, subarbol.altura())
        return altura_actual + 1
    
    def ancho(self, nivel=0, nodos=defaultdict(int)):
        '''Esta función devuelve el nivel del arbol con mayor cantidad de nodos.
        
        Args:
            nivel (int): nivel del árbol. Se inicializa en 0 para poder hacer la recursión.
            nodos (defaultdict(int)): diccionario con la cantidad de nodos en cada nivel.
            
            Returns:
                int: nivel del arbol con mayor cantidad de nodos.'''
        nodos[nivel] += 1
        for subarbol in self.subs:
            subarbol.ancho(nivel + 1, nodos)
        return max(nodos.values())
    
    def ancho_nivel(self, nivel=0, nodos=defaultdict(int)) -> int:
        '''Esta función devuelve la cantidad de nodos por nivel.

        Args:  
            nivel (int): nivel del árbol. Se inicializa en 0 para poder hacer la recursión.
            nodos (defaultdict(int)): diccionario con la cantidad de nodos en cada nivel.

        Returns:    
            int: cantidad de nodos en el nivel dado.
        '''        
        nodos[nivel] += 1
        for subarbol in self.subs:
            subarbol.ancho_nivel(nivel + 1, nodos)
        return nodos[nivel]
    
    def posorden(self) -> list["Arbol"]:
        '''Esta función devuelve una lista con los nodos del árbol en posorden.

        Returns:
            list[Arbol]: lista con los nodos del árbol en posorden.
        '''
        recorrido = []
        for subarbol in self.subs:
            recorrido += subarbol.posorden()
        recorrido.append(self)
        return recorrido
    
    @abstractmethod
    def agregar_subarbol(self, subarbol):
        '''Funcion abstracta que agrega un subarbol al árbol.'''
        raise NotImplementedError
    
    @abstractmethod
    def copy(self):
        '''Funcion abstracta que devuelve una copia profunda del árbol.'''
        raise NotImplementedError
    
    @abstractmethod
    def es_raiz(self):
        '''Funcion abstracta que devuelve True si el nodo es la raiz.'''
        raise NotImplementedError
    
class ArbolClasificador(Arbol, Clasificador, ABC):
    '''Clase abstracta que define los métodos necesarios para un árbol de decisión clasificador.'''
    def __init__(self, **kwargs):
        '''Inicializa un árbol de decisión clasificador.
        
        Args:
            **kwargs: hiperparámetros del árbol.

        Atributos:
            hipermarametros (Hiperparametros): hiperparámetros del árbol.        
            data (pd.DataFrame): datos de entrenamiento.
            target (pd.Series): target de los datos de entrenamiento.
            Los siguientes atributos se inicializan como None y se completan durante el entrenamiento:
            target_categorias (list[str]): categorías del target.
            atributo_split (str): atributo con el que se hizo el split.
            atributo_split_anterior (str): atributo con el que se hizo el split en el nodo anterior.
            valor_split_anterior (str): valor con el que se hizo el split en el nodo anterior.
            signo_split_anterior (str): signo con el que se hizo el split en el nodo anterior.
            clase (str): clase del nodo.
            impureza (Impureza): impureza del nodo.
        '''
        super().__init__()
        hiperparametros = Hiperparametros(**kwargs)  
        for key, value in hiperparametros.__dict__.items():
            setattr(self, key, value)
        self.data: Optional[pd.DataFrame ] = None
        self.target: Optional[pd.Series] = None
        self.target_categorias: Optional[list[str]]= None
        self.atributo_split: Optional[str] = None
        self.atributo_split_anterior: Optional[str] = None
        self.valor_split_anterior: Optional[str]= None
        self.signo_split_anterior: Optional[str] = None
        self.clase: Optional[str] = None
        self.impureza: Optional[Impureza] = None
    
    def es_raiz(self) -> bool:
        '''Función que devuelve True si el nodo es la raiz.
        
        Returns:
            bool: True si el nodo es la raiz, False en caso contrario.'''
        return self.valor_split_anterior is None
    
    def _values(self) -> list[int]:
        '''Esta función devuelve la cantidad de muestras por clase en el target.
        
        Returns:
            list[int]: cantidad de muestras por categoría en el target.'''
        recuento_values = self.target.value_counts()
        values = []
        for valor in self.target_categorias:
            value = recuento_values.get(valor, 0)
            values.append(value)
        return values
    
    def set_clase(self) -> None:
        '''Esta función setea el atributo clase del nodo, se elige la clase con mayor cantidad de muestras.'''
        if len(self.target) != 0:
            self.clase = self.target.value_counts().idxmax()
        else:
            ValueError("No hay valores en el target")
        

    def set_target_categorias(self, y) -> None:
        '''Esta función setea el atributo que contiene las categorías del target.'''
        self.target_categorias = y.unique()
    
    def _total_samples(self) -> int:
        '''Esta función devuelve la cantidad de muestras en el nodo.'''
        return len(self.data)
    
    def agregar_subarbol(self, subarbol) -> None:
        '''Esta función agrega un subarbol al árbol.

        Args:
            subarbol (ArbolClasificador): subarbol a agregar.
        '''
        for key, value in self.__dict__.items():
            if key in Hiperparametros().__dict__:  
                setattr(subarbol, key, value)
        self.subs.append(subarbol)
    
    def graficar(self) -> None:
        '''Esta función grafica el árbol de decisión.'''
        if self.data is None or self.target is None:
            raise ArbolNoEntrenadoException()
        graficador = GraficadorArbol(self)
        graficador.graficar()
    
    def graficar_feo(self) -> None:
        '''Esta función grafica el árbol de decisión.'''
        if self.data is None or self.target is None:
            raise ArbolNoEntrenadoException()
        graficador = GraficadorArbolFeo(self)
        graficador.plot()

    def reduced_error_pruning(self, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        '''Esta función realiza la poda del arbol con el metodo Reduced Error Pruning.
        
        Args:
            x_test (pd.DataFrame): datos de test.
            y_test (pd.Series): target de los datos de test.
        '''
        if self.data is None or self.target is None:
            raise ArbolNoEntrenadoException()
        def _interna_rep(arbol: ArbolClasificador, x_test, y_test):
            if not arbol.es_hoja():
                for subarbol in arbol.subs:
                    _interna_rep(subarbol, x_test, y_test)

                pred_raiz: list[str] = arbol.predict(x_test)
                error_clasif_raiz = Metricas.error_score(y_test, pred_raiz)

                error_clasif_ramas = 0.0

                for subarbol in arbol.subs:
                    pred_podada = subarbol.predict(x_test) 
                    error_clasif_podada = Metricas.error_score(y_test, pred_podada)
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
        '''Esta función realiza la poda del arbol con el metodo Reduced Error Pruning.
        
        Args:
            x_val (pd.DataFrame): datos de validación.
            y_val (pd.Series): target de los datos de validación.
            margen (float): margen de error permitido para realizar la poda.
        '''
        if self.data is None or self.target is None:
            raise ArbolNoEntrenadoException()
        arbol_completo = deepcopy(self)
        error_inicial = Metricas.error_score(y_val, arbol_completo.predict(x_val))
        nodos = arbol_completo.posorden()
        for nodo in nodos:
            if not nodo.es_hoja():
                arbol_podado = arbol_completo._podar(nodo)
                nuevo_error = Metricas.error_score(y_val, arbol_podado.predict(x_val))
                if nuevo_error - margen < error_inicial:
                    arbol_completo = arbol_podado
                    error_inicial = nuevo_error
        return arbol_completo

    @abstractmethod
    def _mejor_atributo_split(self) -> str | None:
        '''Esta función abstracta devuelve el mejor atributo para hacer el split.

        Returns:
            str: mejor atributo para hacer el split.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str) -> None:
        '''Esta función abstracta realiza el split del árbol.

        Args:
            atributo (str): atributo con el que se hace el split.
        '''
        raise NotImplementedError

class Bosque(ABC):
    '''Clase abstracta que define los métodos necesarios para un bosque de árboles.'''
    def __init__(self, cantidad_arboles: int = 10) -> None:
        '''Inicializa un bosque de árboles.

        Args:
            cantidad_arboles (int): cantidad de árboles en el bosque.

        Atributos:
            arboles (list[Arbol]): lista con los árboles del bosque.
            cantidad_arboles (int): cantidad de árboles en el bosque.
        '''
        self.arboles: list[Arbol] = []
        self.cantidad_arboles = cantidad_arboles