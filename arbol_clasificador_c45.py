from copy import deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional
from graficador import GraficadorArbol
from _superclases import ArbolClasificador, Hiperparametros
from metricas import Metricas


class ArbolDecisionC45(ArbolClasificador):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.umbral_split: Optional[float] = None

    def copy(self):
        nuevo = ArbolDecisionC45(**self.__dict__)
        nuevo.data = self.data.copy()
        nuevo.target = self.target.copy()
        nuevo.target_categorias = self.target_categorias.copy()
        nuevo.subs = [sub.copy() for sub in self.subs]
        return nuevo
    
    def agregar_subarbol(self, subarbol):
        for key, value in self.__dict__.items():
            if key in Hiperparametros().__dict__:  # Solo copiar los atributos que están en Hiperparametros
                setattr(subarbol, key, value)
        self.subs.append(subarbol)
        
    def _nuevo_subarbol(self, atributo: str, operacion: str, valor: Any):
        nuevo = ArbolDecisionC45(**self.__dict__)
        if operacion == "menor":
            nuevo.data = self.data[self.data[atributo] < valor]
            nuevo.target = self.target[self.data[atributo] < valor]
        
        elif operacion == "mayor":
            nuevo.data = self.data[self.data[atributo] > valor]
            nuevo.target = self.target[self.data[atributo] > valor]
        
        elif operacion == "igual":
            nueva_data = self.data[self.data[atributo] == valor]
            nueva_data = nueva_data.drop(atributo, axis=1)
            nuevo.data = nueva_data
            nuevo.target = self.target[self.data[atributo] == valor]
        
        nuevo.clase = nuevo.target.value_counts().idxmax()
        nuevo.atributo_split_anterior = atributo
        nuevo.valor_split_anterior = valor
        self.agregar_subarbol(nuevo)
        
    def _split_numerico(self, atributo: str, umbral: float | int) -> None:
        self.atributo_split = atributo
        self.umbral_split = umbral
        self._nuevo_subarbol(atributo, "menor", umbral)
        self._nuevo_subarbol(atributo, "mayor", umbral)

    def _split_categorico(self, atributo: str) -> None:
        self.atributo_split = atributo
        for categoria in self.data[atributo].unique():
            self._nuevo_subarbol(atributo, "igual", categoria)
    
    # respeta la firma de la superclase (o de id3 en caso de decidir que sea subclase)
    def _split(self, atributo):
        if self.es_atributo_numerico(atributo):
            self.tipo_atributo = "C" #Continuo   # guarda el tipo de atributo por el que se hace split
            mejor_umbral = self._mejor_umbral_split(atributo)
            self._split_numerico(atributo, mejor_umbral)
        else:
            self.tipo_atributo = "G" #Categorico  # guarda el tipo de atributo por el que se hace split
            self._split_categorico(atributo)

    def es_atributo_numerico(self, atributo: str) -> bool:
        return pd.api.types.is_numeric_dtype(self.data[atributo])
    
    def _entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize=True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia if entropia != 0 else 0
    
    def _information_gain_base(self, atributo: str, split: Callable):
        entropia_actual = self._entropia()
        len_actual = len(self.data)
        nuevo = deepcopy(self) # usar copy propio cuando funcione

        split(nuevo, atributo)

        entropias_subarboles = 0 
        for subarbol in nuevo.subs:
            entropia = subarbol._entropia()
            len_subarbol = len(subarbol.data)
            entropias_subarboles += ((len_subarbol/len_actual) * entropia)

        information_gain = entropia_actual - entropias_subarboles
        return information_gain
    
    def _information_gain(self, atributo: str) -> float:  #IMPORTANTE: este information gain calcula el mejor umbral de ser necesario
        def split(arbol, atributo):
            arbol._split(atributo)
        
        return self._information_gain_base(atributo, split)
    
    def _split_info(self):
        split_info = 0
        len_actual = len(self.data)
        for subarbol in self.subs:
            len_subarbol = len(subarbol.data)
            split_info += (len_subarbol / len_actual) * np.log2(len_subarbol / len_actual)
        return -split_info
        
    def _gain_ratio(self, atributo: str) -> float:
        nuevo = deepcopy(self) # usar copy propio

        information_gain = nuevo._information_gain(atributo)
        nuevo._split(atributo)
        split_info = nuevo._split_info()

        return information_gain / split_info
    
    def _mejor_atributo_split(self) -> str | None:
        mejor_gain_ratio = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            if len(self.data[atributo].unique()) > 1:
                gain_ratio = self._gain_ratio(atributo)
                if gain_ratio > mejor_gain_ratio:
                    mejor_gain_ratio = gain_ratio
                    mejor_atributo = atributo

        return mejor_atributo
    
    def __information_gain_numerico(self, atributo: str, umbral: float | int):  # helper de mejor_umbral_split, no calcula el mejor umbral
            def split_num(arbol, atributo):
                arbol._split_numerico(atributo, umbral)
            
            return self._information_gain_base(atributo, split_num) # clausura, se deberia llevar el umbral
    
    def _mejor_umbral_split(self, atributo: str) -> float:
        
        self.data = self.data.sort_values(by=atributo)
        mejor_ig = -1
        valores_unicos = self.data[atributo].unique()
        mejor_umbral = valores_unicos[0]

        i = 0
        while i < len(valores_unicos) - 1:
            umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
            ig = self.__information_gain_numerico(atributo, umbral) # uso information_gain, gain_ratio es para la seleccion de atributo
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_umbral = umbral
            i += 1

        return mejor_umbral
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target = y
        self.data = X
        self.set_clase()

        def _interna(arbol: ArbolDecisionC45, prof_acum: int = 1):
            arbol.set_target_categorias(y)

            if arbol._puede_splitearse(prof_acum):
                mejor_atributo = arbol._mejor_atributo_split()

                if mejor_atributo:
                    arbol._split(mejor_atributo) # el check de numerico ahora ocurre dentro de _split()
                    
                    for sub_arbol in arbol.subs:
                        _interna(sub_arbol, prof_acum + 1)
        
        _interna(self)
        
    def predict(self, X: pd.DataFrame) -> list:
        predicciones = []
        def _recorrer(arbol, fila: pd.Series) -> None:
            if arbol.es_hoja():
                predicciones.append(arbol.clase)
            else:
                valor = fila[arbol.atributo_split]
                if arbol.es_atributo_numerico(arbol.atributo_split):  # es split numerico
                    if valor < arbol.umbral_split:
                        _recorrer(arbol.subs[0], fila)
                    elif valor > arbol.umbral_split:
                        _recorrer(arbol.subs[1], fila)
                else:
                    for subarbol in arbol.subs:
                        if valor == subarbol.valor_split_anterior:
                            _recorrer(subarbol, fila)
        
        for _, fila in X.iterrows():
            _recorrer(self, fila)
        return predicciones
    
    def graficar(self):    
        plotter = GraficadorArbol(self)
        plotter.plot()

    def _error_clasificacion(self, y, y_pred):
        x = []
        for i in range(len(y)):
            x.append(y[i] != y_pred[i])
        return np.mean(x)
        
    def Reduced_Error_Pruning(self, x_test: Any, y_test: Any):
            def _interna_REP(arbol: ArbolDecisionC45, x_test, y_test):
                if arbol.es_hoja():
                    return

                for subarbol in arbol.subs:
                    _interna_REP(subarbol, x_test, y_test)

                    pred_raiz: list[str] = arbol.predict(x_test)
                    accuracy_raiz = Metricas.accuracy_score(y_test.tolist(), pred_raiz)
                    error_clasif_raiz = arbol._error_clasificacion(y_test.tolist(), pred_raiz)

                    error_clasif_ramas = 0.0

                    for rama in arbol.subs:
                        new_arbol: ArbolDecisionC45 = rama
                        pred_podada = new_arbol.predict(x_test)
                        accuracy_podada = Metricas.accuracy_score(y_test.tolist(), pred_podada)
                        error_clasif_podada = new_arbol._error_clasificacion(y_test.tolist(), pred_podada)
                        error_clasif_ramas = error_clasif_ramas + error_clasif_podada

                    if error_clasif_ramas < error_clasif_raiz:
                        #print(" * Podar \n")
                        arbol.subs = []
                    #else:
                        #print(" * No podar \n")

            _interna_REP(self, x_test, y_test)
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        
        if self.es_atrib_continuo():  # self.es_atributo_numerico(self.atributo_split) #TODO
            simbolo_rama = '└─ NO ── ' if es_ultimo else '├─ SI ── '
            split = f"{self.atributo_split} < {self.umbral_split:.2f} ?" if self.umbral_split else ""
        else:
            simbolo_rama = '└─── ' if es_ultimo else '├─── '
            split = "Split: " + str(self.atributo_split)
        
        entropia = f"Entropia: {self._entropia():.2f}"
        samples = f"Samples: {self._total_samples()}"
        values = f"Values: {self._values()}"
        clase = f"Clase: {self.clase}"

        if self.es_raiz():
            print(entropia)
            print(samples)
            print(values)
            print(clase)
            print(split)

            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)

        elif not self.es_hoja():
            print(prefijo + "│")
            
            if self.es_atrib_continuo():
                print(prefijo + simbolo_rama + entropia)
                prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                print(prefijo2 + samples)
                print(prefijo2 + values)
                print(prefijo2 + clase)
                print(prefijo2 + split)
            else:
                rta =  f"{self.atributo_split_anterior} = {self.valor_split_anterior}"
                print(prefijo + simbolo_rama + rta)            
                prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                print(prefijo2 + entropia)
                print(prefijo2 + samples)
                print(prefijo2 + values)
                print(prefijo2 + clase)
                print(prefijo2 + split)

            prefijo += ' ' * 10 if es_ultimo else '│' + ' ' * 9
            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)

        else: # es hoja
            prefijo_hoja = prefijo + " " * len(simbolo_rama) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
            print(prefijo + "│")
                        
            if self.es_atrib_continuo(): # nunca entra aca porque una hoja nunca hace split
                print(prefijo + simbolo_rama + entropia)
                print(prefijo_hoja + samples)
                print(prefijo_hoja + values)
                print(prefijo_hoja + clase)
            else:
                rta =  f"{self.atributo_split_anterior} = {self.valor_split_anterior}"
                print(prefijo + simbolo_rama + rta)            
                print(prefijo_hoja + entropia)
                print(prefijo_hoja + samples)
                print(prefijo_hoja + values)
                print(prefijo_hoja + clase)    
                
def probar(df, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    arbol = ArbolDecisionC45()
    arbol.fit(x_train, y_train)
    arbol.imprimir()
    y_pred = arbol.predict(x_test)

    #arbol.Reduced_Error_Pruning(x_test, y_test)

    print(f"\naccuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
    print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")


if __name__ == "__main__":
    import sklearn.datasets

    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    print("pruebo con iris")
    probar(df, "target")

    print("pruebo con tennis")
    tennis = pd.read_csv("./datasets/PlayTennis.csv")

    probar(tennis, "Play Tennis")


    # TODO: arreglar el predict, hay un issue
    # print("pruebo con patients") 

    # patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    # patients = patients.drop("Patient Id", axis = 1)
    # patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str) # para que sean categorias
    
    # probar(patients, "Level")
    