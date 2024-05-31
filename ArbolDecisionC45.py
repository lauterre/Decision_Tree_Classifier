from copy import deepcopy
from sklearn.model_selection import train_test_split
from ArbolDecisionID3 import ArbolDecisionID3
import pandas as pd
import numpy as np
from typing import Any, Optional

from Metricas import Metricas


class ArbolDecisionC45(ArbolDecisionID3):
    def __init__(self, **kwargs) -> None:
        try:
            super().__init__(**kwargs)
            self.valor_split: Optional[float] = None
        except Exception as e:
            print(f"Error al inicializar el ArbolDecisionC45: {e}")

    def _nuevo_subarbol(self, atributo: str, operacion: str, valor: Any):
        try:
            nuevo = ArbolDecisionC45()
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
        except Exception as e:
            print(f"Error al tratar de generar un nuevo subarbol: {e}")

    def _split(self, atributo: str, valor: Any = None) -> None:
        try:
            self.atributo_split = atributo
            self.valor_split = valor
            if valor:
                self._nuevo_subarbol(atributo, "menor", valor)
                self._nuevo_subarbol(atributo, "mayor", valor)
            else:
                for categoria in self.data[atributo].unique():
                    self._nuevo_subarbol(atributo, "igual", categoria)
        except Exception as e:
            print(f"Error al tratar de dividir el arbol por el atributo '{atributo}': {e}")

    # No me gusta esto de pasar None y tampoco me gusta no sobreescribir los metodos (_information_gain_continuo y demás)
    # preguntar a Mariano
    def _information_gain(self, atributo: str, valor=None) -> float:
        try:
            # si valor no es none estamos usando un atributo numerico
            if valor:
                entropia_actual = self._entropia()
                len_actual = len(self.data)

                information_gain = entropia_actual

                nuevo = deepcopy(self)
                nuevo._split(atributo, valor)

                entropia_izq = nuevo.subs[0]._entropia()
                len_izq = len(nuevo.subs[0].data)
                entropia_der = nuevo.subs[1]._entropia()
                len_der = len(nuevo.subs[1].data)

                information_gain -= ((len_izq / len_actual) * entropia_izq + (len_der / len_actual) * entropia_der)
            else: # si no es continuo
                information_gain = super()._information_gain(atributo)
            return information_gain
        except Exception as e:
            print(f"Error al tratar de calcular el information gain para el atributo '{atributo}': {e}")
            return 0

    def _split_info(self):
        try:
            split_info = 0
            len_actual = len(self.data)
            for subarbol in self.subs:
                len_subarbol = len(subarbol.data)
                split_info += (len_subarbol / len_actual) * np.log2(len_subarbol / len_actual)
            return -split_info
        except Exception as e:
            print(f"Error al tratar de calcular el split info: {e}")
            return 0

    def _gain_ratio(self, atributo: str):
        try:
            nuevo = deepcopy(self)

            information_gain = nuevo._information_gain(atributo)
            umbral = None

            if pd.api.types.is_numeric_dtype(self.data[atributo]): #es numerico
                umbral = nuevo._mejor_umbral_split(atributo)

            nuevo._split(atributo, umbral)
            split_info = nuevo._split_info()

            return information_gain / split_info
        except Exception as e:
            print(f"Error al calcular el gain ratio para el atributo '{atributo}': {e}")
            return 0

    def _mejor_atributo_split(self) -> str:
        try:
            mejor_gain_ratio = -1
            mejor_atributo = None
            atributos = self.data.columns

            for atributo in atributos:
                gain_ratio = self._gain_ratio(atributo)
                if gain_ratio > mejor_gain_ratio:
                    mejor_gain_ratio = gain_ratio
                    mejor_atributo = atributo

            return mejor_atributo
        except Exception as e:
            print(f"Error al tratar de encontrar el mejor atributo para realizar el split: {e}")
            return ""

    def _mejor_umbral_split(self, atributo: str) -> float:
        try:
            self.data = self.data.sort_values(by=atributo)

            mejor_ig = -1
            mejor_umbral = None

            valores_unicos = self.data[atributo].unique()

            i = 0
            while i < len(valores_unicos) - 1:
                umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
                ig = self._information_gain(atributo, umbral) # uso information_gain, gain_ratio es para la seleccion de atributo
                if ig > mejor_ig:
                    mejor_ig = ig
                    mejor_umbral = umbral
                i += 1

            return float(mejor_umbral)
        except Exception as e:
            print(f"Error al tratar de encontrar el mejor threshold para el atributo '{atributo}': {e}")
            return 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.target = y
            self.data = X
            self.clase = self.target.value_counts().idxmax()

            def _interna(arbol: ArbolDecisionC45, prof_acum: int = 0):
                try:
                    arbol.target_categorias = y.unique()

                    if prof_acum == 0:
                        prof_acum = 1

                    if not (len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0
                            or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum)
                            or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples())):

                        mejor_atributo = arbol._mejor_atributo_split()

                        if pd.api.types.is_numeric_dtype(self.data[mejor_atributo]): # si es numerica
                            mejor_umbral = arbol._mejor_umbral_split(mejor_atributo)
                            arbol._split(mejor_atributo, mejor_umbral)
                        else:
                            arbol._split(mejor_atributo)

                        for sub_arbol in arbol.subs:
                            _interna(sub_arbol, prof_acum + 1)
                except Exception as e:
                    print(f"Error en la función interna del fit: {e}")

            _interna(self)
        except Exception as e:
            print(f"Error al tratar de fittear el arbol: {e}")

    def predict(self, X: pd.DataFrame) -> list:
        predicciones = []
        try:
            def _recorrer(arbol, fila: pd.Series) -> None:
                try:
                    if arbol.es_hoja():
                        predicciones.append(arbol.clase)
                    else:
                        valor = fila[arbol.atributo_split]
                        if pd.api.types.is_numeric_dtype(valor):
                            if valor < arbol.valor_split:
                                _recorrer(arbol.subs[0], fila)
                            elif valor > arbol.valor_split:
                                _recorrer(arbol.subs[1], fila)
                        else:
                            for subarbol in arbol.subs:
                                if valor == subarbol.valor_split_anterior:
                                    _recorrer(subarbol, fila)
                except Exception as e:
                    print(f"Error de predicción para la fila: {e}")

            for _, fila in X.iterrows():
                _recorrer(self, fila)
        except Exception as e:
            print(f"Error de predicción: {e}")
        return predicciones
    
    # TODO: adaptar para los split categoricos
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        try:
            simbolo_rama = '└─ NO ── ' if es_ultimo else '├─ SI ── '
            split = f"{self.atributo_split} < {self.valor_split:.2f} ?" if self.valor_split else ""
            #umbral = f"Umbral: {self.valor_split}"
            #rta = f"{self.atributo_split_anterior} > {self.valor_split_anterior}" if es_ultimo else f"{self.atributo_split_anterior} < {self.valor_split_anterior}"
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
                #print(prefijo + simbolo_rama + rta)
                print(prefijo + simbolo_rama + entropia)
                prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                #print(prefijo2 + entropia)
                print(prefijo2 + samples)
                print(prefijo2 + values)
                print(prefijo2 + clase)
                print(prefijo2 + split)
                #print(prefijo2 + umbral)

                prefijo += ' ' * 10 if es_ultimo else '│' + ' ' * 9
                for i, sub_arbol in enumerate(self.subs):
                    ultimo: bool = i == len(self.subs) - 1
                    sub_arbol.imprimir(prefijo, ultimo)
            else:
                prefijo_hoja = prefijo + " " * len(simbolo_rama) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                print(prefijo + "│")
                #print(prefijo + simbolo_rama + rta)
                print(prefijo + simbolo_rama + entropia)
                #print(prefijo_hoja + entropia)
                print(prefijo_hoja + samples)
                print(prefijo_hoja + values)
                print(prefijo_hoja + clase)
        except Exception as e:
            print(f"Error al trata de imprimir por pantalla el arbol: {e}")

def probar(df, target: str):
    try:
        X = df.drop(target, axis=1)
        y = df[target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        arbol = ArbolDecisionC45(min_obs_nodo=1)
        arbol.fit(x_train, y_train)
        arbol.imprimir()
        y_pred = arbol.predict(x_test)

        arbol.Reduced_Error_Pruning(x_test, y_test)

        print(f"\naccuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
        print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")
    except Exception as e:
        print(f"Error de testeo del modelo: {e}")


if __name__ == "__main__":
    import sklearn.datasets
    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    X = df.drop("target", axis = 1)
    y = df["target"]
    
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    arbol = ArbolDecisionC45(max_prof=5) #TODO: no anda el fit cuando el max_prof es mayot (o no hay limite de profundidad)
    arbol.fit(X_train, y_train)
    arbol.imprimir()
    arbol.graficar() 
    y_pred = arbol.predict(x_test)

    print(f"\naccuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
    print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio= "macro"):.2f}\n")

    # print("pruebo con patients") 

    # patients = pd.read_csv("cancer_patients.csv", index_col=0)
    # patients = patients.drop("Patient Id", axis = 1)

    # X = patients.drop("Level", axis = 1)
    # y = patients["Level"]
    # patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str) # para que sean categorias
    
    # X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # arbol = ArbolDecisionC45(max_prof=4)
    # arbol.fit(X_train, y_train)
    # # arbol.imprimir() no funciona
    # y_pred = arbol.predict(x_test)

    # print(f"\naccuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
    # print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio= "ponderado"):.2f}\n")

    print("pruebo con tennis")

    tennis = pd.read_csv("PlayTennis.csv")

    X = tennis.drop("Play Tennis", axis = 1)
    y = tennis["Play Tennis"]
    
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    arbol = ArbolDecisionC45(max_prof=2)
    arbol.fit(X_train, y_train)
    # arbol.imprimir() no funciona
    y_pred = arbol.predict(x_test)

    print(f"\naccuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
    print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio= "micro"):.2f}\n")