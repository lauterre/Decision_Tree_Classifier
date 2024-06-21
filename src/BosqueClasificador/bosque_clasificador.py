import pandas as pd
import numpy as np
from src.ArbolDecision.arbol_clasificador_C45 import ArbolClasificadorC45
from src.Superclases.superclases import Clasificador, Bosque, Hiperparametros
from src.ArbolDecision.arbol_clasificador_ID3 import ArbolClasificadorID3
from src.Excepciones.excepciones import BosqueEntrenadoException, BosqueNoEntrenadoException


class BosqueClasificador(Bosque, Clasificador):
    '''Clase que representa un bosque de árboles clasificadores.'''
    def __init__(self, clase_arbol: str = "id3", cantidad_arboles: int = 10, cantidad_atributos:str ='sqrt',verbose: bool = False,**kwargs) -> None:
        '''Constructor de la clase BosqueClasificador.

        Args:
            clase_arbol (str): Clase de árbol a utilizar. Puede ser 'id3' o 'c45'.
            cantidad_arboles (int): Cantidad de árboles a construir.
            cantidad_atributos (str): Cantidad de atributos a considerar en cada árbol. Puede ser 'all', 'log2', 'sqrt'.
            verbose (bool): Indica si se imprimen mensajes durante el entrenamiento.
            **kwargs: Hiperparámetros del árbol.
        '''
        super().__init__(cantidad_arboles)

        hiperparametros = {k: v for k, v in kwargs.items() if k in Hiperparametros.PARAMS_PERMITIDOS}
        self.hiperparametros_arbol = Hiperparametros(**hiperparametros)
        
        for key, value in self.hiperparametros_arbol.__dict__.items():
            setattr(self, key, value)
        self.cantidad_atributos = cantidad_atributos
        self.clase_arbol = clase_arbol
        self.verbose = verbose

    def _bootstrap_samples(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        '''Genera un conjunto de muestras de entrenamiento a partir de X e y.

        Args:
            X (pd.DataFrame): Conjunto de datos de entrenamiento.
            y (pd.Series): Etiquetas de los datos de entrenamiento.

        Returns:
            pd.DataFrame: Muestras de entrenamiento generadas.
            pd.Series: Etiquetas de las muestras de entrenamiento generadas.
        '''
        n_samples = X.shape[0]
        atributos = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[atributos].reset_index(drop=True), y.iloc[atributos].reset_index(drop=True)

    def seleccionar_atributos(self, X: pd.DataFrame)-> list[int]:
        '''
        Selecciona aleatoriamente los atributos con los que se va a entrenar el arbol.
        El atributo cantidad_atributos indica la cantidad de atributos a seleccionar. 

        Args:
            X (pd.DataFrame): Conjunto de datos de entrenamiento.

        Returns:
            list[int]: indices de los atributos seleccionados
        '''
        n_features = X.shape[1]
        if self.cantidad_atributos == 'all':
            size = n_features
        elif self.cantidad_atributos == 'log2':
            size = int(np.log2(n_features))
        elif self.cantidad_atributos == 'sqrt':
            size = int(np.sqrt(n_features))
        else:
            raise ValueError("cantidad_atributos debe ser 'all', 'log2' o 'sqrt'")

        indices = np.random.choice(n_features, size, replace=False)
        return indices

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''Entrena el bosque de árboles clasificadores.

        Args:
            X (pd.DataFrame): Conjunto de datos de entrenamiento.
            y (pd.Series): Etiquetas de los datos de entrenamiento.
        '''
        if self.arboles:
            raise BosqueEntrenadoException()
        for _ in range(self.cantidad_arboles):
            if self.verbose : print(f"Contruyendo arbol nro: {_ + 1}") 
            # Bootstrapping
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Selección de atributos
            atributos = self.seleccionar_atributos(X_sample)
            X_sample = X_sample.iloc[:, atributos]

            # Crear y entrenar un nuevo árbol
            if self.clase_arbol == 'id3':
                arbol = ArbolClasificadorID3(**self.hiperparametros_arbol.__dict__)
                arbol.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                self.arboles.append(arbol)
            elif self.clase_arbol == 'c45':
                arbol = ArbolClasificadorC45(**self.hiperparametros_arbol.__dict__)
                arbol.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                self.arboles.append(arbol)
            else:
                raise ValueError("Clase de arbol soportado por el bosque: 'id3', 'c45'")
            #arbol.imprimir()

    def predict(self, X: pd.DataFrame) -> list:
        '''Realiza predicciones sobre un conjunto de datos.

        Args:
            X (pd.DataFrame): Conjunto de datos de prueba.

        Returns:    
            predicciones_finales (list): Predicciones realizadas.
        '''
        if not self.arboles:
            raise BosqueNoEntrenadoException()
        todas_predicciones = pd.DataFrame(index=X.index, columns=range(len(self.arboles))) 
        
        for i, arbol in enumerate(self.arboles):
            todas_predicciones[i] = arbol.predict(X)

        # Aplicar la votación mayoritaria
        predicciones_finales = todas_predicciones.apply(lambda x: x.value_counts().idxmax(), axis=1)
        
        return list(predicciones_finales)
