import pandas as pd
from src.Excepciones.excepciones import ArbolEntrenadoException, ArbolNoEntrenadoException
from src.Impureza.impureza import Entropia
from src.Superclases.superclases import ArbolClasificador, Hiperparametros

class ArbolClasificadorID3(ArbolClasificador):
    '''Clase que representa un árbol de decisión que utiliza el algoritmo ID3 para clasificar.'''
    def __init__(self, **kwargs) -> None:
        '''Constructor de la clase ArbolClasificadorID3. Hereda de ArbolClasificador.
        Args:
            **kwargs: hiperparametros del árbol.
        '''
        super().__init__(**kwargs)
        self.impureza = Entropia()

    def copy(self) -> "ArbolClasificadorID3":
        '''Devuelve una copia profunda del arbol
        
        Returns:
            ArbolClasificadorID3: copia del arbol'''
        hiperparametros_copiados = {k: v for k, v in self.__dict__.items() if k in Hiperparametros.PARAMS_PERMITIDOS}
        nuevo = ArbolClasificadorID3(**hiperparametros_copiados)
        nuevo.data = self.data.copy()
        nuevo.target = self.target.copy()
        nuevo.target_categorias = self.target_categorias.copy()
        nuevo.set_clase()
        nuevo.atributo_split = self.atributo_split
        nuevo.atributo_split_anterior = self.atributo_split_anterior
        nuevo.valor_split_anterior = self.valor_split_anterior
        nuevo.signo_split_anterior = self.signo_split_anterior
        nuevo.impureza = self.impureza
        nuevo.subs = [sub.copy() for sub in self.subs]
        return nuevo
    
    def _mejor_atributo_split(self) -> str | None:
        '''Calcula el mejor atributo por el cual splitear el arbol.
        
        Returns:
            str: nombre del mejor atributo por el cual splitear el arbol.'''
        mejor_ig = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            if len(self.data[atributo].unique()) > 1: 
                ig = self._information_gain(atributo)
                if ig > mejor_ig:
                    mejor_ig = ig
                    mejor_atributo = atributo

        return mejor_atributo
        
    def _split(self, atributo: str) -> None:
        '''Realiza el split del arbol por el atributo dado.

        Args:
            atributo (str): atributo por el cual splitear el arbol.
        '''
        
        self.atributo_split = atributo  # guardo el atributo por el cual spliteo

        for categoria in self.data[atributo].unique():  # recorre el dominio de valores del atributo
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis=1)  # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]

            nuevo_arbol = ArbolClasificadorID3()  # Crea un nuevo arbol
            nuevo_arbol.data = nueva_data  # Asigna nodo
            nuevo_arbol.target = nuevo_target  # Asigna target
            nuevo_arbol.valor_split_anterior = categoria
            nuevo_arbol.atributo_split_anterior = atributo
            nuevo_arbol.set_clase()
            nuevo_arbol.signo_split_anterior = '='
            self.agregar_subarbol(nuevo_arbol)
            
    def _information_gain(self, atributo: str) -> float:
        '''Calcula la ganancia de información de splitear por el atributo dado.

        Args:  
            atributo (str): atributo por el cual splitear el arbol.

        Returns:
            float: ganancia de información de splitear por el atributo dado.
        '''
        def split(arbol, atributo):
            arbol._split(atributo)

        return self.impureza.calcular_impureza_split(self, atributo, split) # quizas renombrar a ganancia (o evaluar_split) en impureza     

    def _puede_splitearse(self, prof_acum: int, mejor_atributo: str) -> bool:
        '''Verifica si el arbol puede splitearse por el atributo dado.

        Args:
            prof_acum (int): profundidad acumulada del arbol.
            mejor_atributo (str): atributo por el cual splitear el arbol.

        Returns:
            bool: True si el arbol puede splitearse, False en caso contrario.
        '''
        copia = self.copy()
        information_gain = self._information_gain(mejor_atributo)
        copia._split(mejor_atributo)
        for subarbol in copia.subs:
            if self.min_obs_hoja != -1 and subarbol._total_samples() < self.min_obs_hoja:
                return False
            
        return not (len(self.target.unique()) == 1 or len(self.data.columns) == 0
                    or (self.max_prof != -1 and self.max_prof <= prof_acum)
                    or (self.min_obs_nodo != -1 and self.min_obs_nodo > self._total_samples())
                    or (self.min_infor_gain != -1 and self.min_infor_gain > information_gain))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''Entrena el arbol de decisión con los datos de entrada.

        Args:
            X (pd.DataFrame): datos de entrenamiento.
            y (pd.Series): vector con el atributo a predecir.
        '''
        if self.data is not None and self.target is not None:
            raise ArbolEntrenadoException()
        self.target = y.copy()
        self.data = X.copy()
        self.set_clase()

        def _interna(arbol, prof_acum: int = 1):
            arbol.set_target_categorias(y)

            mejor_atributo = arbol._mejor_atributo_split()
            if mejor_atributo and arbol._puede_splitearse(prof_acum, mejor_atributo):
                arbol._split(mejor_atributo)
                
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum + 1)
        
        _interna(self)

    def predict(self, X: pd.DataFrame) -> list[str]:
        '''Predice la clase de las instancias de entrada. 

        Args:
            X (pd.DataFrame): instancias a predecir.

        Returns:
            predicciones (list): lista con las predicciones.
        '''
        if self.data is None or self.target is None:
            raise ArbolNoEntrenadoException()
        predicciones = []
        def _recorrer(arbol, fila: pd.Series) -> None:
            if arbol.es_hoja():
                predicciones.append(arbol.clase)
            else:
                direccion = fila[arbol.atributo_split]
                existe = False
                for subarbol in arbol.subs:
                    if direccion == subarbol.valor_split_anterior:
                        existe = True
                        _recorrer(subarbol, fila)
                if not existe:
                    predicciones.append(arbol.clase)

        for _, fila in X.iterrows():
            _recorrer(self, fila)

        return predicciones
        
    def __str__(self) -> str:
        if self.data is None or self.target is None:
            raise ArbolNoEntrenadoException()
        out = []
        def _interna(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
            simbolo_rama = '└─── ' if es_ultimo else '├─── '
            split = "Split: " + str(self.atributo_split)
            rta =  f"{self.atributo_split_anterior} = {self.valor_split_anterior}"
            impureza = f"{self.impureza}: {round(self.impureza.calcular(self.target), 3)}"
            samples = f"Muestras: {str(self._total_samples())}"
            values = f"Conteo: {str(self._values())}"
            clase = 'Clase: ' + str(self.clase)
            
            if self.es_raiz():
                out.append(impureza)
                out.append(samples)
                out.append(values)
                out.append(clase)
                out.append(split)

                for i, sub_arbol in enumerate(self.subs):
                    ultimo: bool = i == len(self.subs) - 1
                    _interna(sub_arbol, prefijo, ultimo)

            elif not self.es_hoja():
                out.append(prefijo + "│")
                out.append(prefijo + simbolo_rama + rta)
                prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
                out.append(prefijo2 + impureza)
                out.append(prefijo2 + samples)
                out.append(prefijo2 + values)
                out.append(prefijo2 + clase)
                out.append(prefijo2 + split)

                prefijo += ' ' * 10 if es_ultimo else '│' + ' ' * 9
                for i, sub_arbol in enumerate(self.subs):
                    ultimo: bool = i == len(self.subs) - 1
                    _interna(sub_arbol, prefijo, ultimo)
            else:
                prefijo_hoja = prefijo + " " * len(simbolo_rama) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                out.append(prefijo + "│")
                out.append(prefijo + simbolo_rama + rta)
                out.append(prefijo_hoja + impureza)
                out.append(prefijo_hoja + samples)
                out.append(prefijo_hoja + values)
                out.append(prefijo_hoja + clase)
        _interna(self)
        return "\n".join(out)