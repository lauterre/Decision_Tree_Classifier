from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Any, Optional
from _impureza import Entropia
from _superclases import ArbolClasificador
from metricas import Metricas


class ArbolClasificadorC45(ArbolClasificador):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.umbral_split: Optional[float] = None
        self.impureza = Entropia()

    def copy(self):
        nuevo = ArbolClasificadorC45(**self.__dict__)
        nuevo.data = self.data.copy()
        nuevo.target = self.target.copy()
        nuevo.target_categorias = self.target_categorias.copy()
        nuevo.subs = [sub.copy() for sub in self.subs]
        return nuevo
        
    def _nuevo_subarbol(self, atributo: str, operacion: str, valor: Any):
        nuevo = ArbolClasificadorC45(**self.__dict__)
        if operacion == "menor":
            nuevo.data = self.data[self.data[atributo] < valor]
            nuevo.target = self.target[self.data[atributo] < valor]
            nuevo.signo_split_anterior = "<"
        
        elif operacion == "mayor":
            nuevo.data = self.data[self.data[atributo] >= valor]
            nuevo.target = self.target[self.data[atributo] >= valor]
            nuevo.signo_split_anterior = ">="
        
        elif operacion == "igual":
            nueva_data = self.data[self.data[atributo] == valor]
            nueva_data = nueva_data.drop(atributo, axis=1)
            nuevo.data = nueva_data
            nuevo.target = self.target[self.data[atributo] == valor]
            nuevo.signo_split_anterior = "="
        
        nuevo.set_clase()
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
    
    def _split(self, atributo: str):
        if self.es_atributo_numerico(atributo):
            mejor_umbral = self._mejor_umbral_split(atributo)
            self._split_numerico(atributo, mejor_umbral)
        else:
            self._split_categorico(atributo)

    def es_atributo_numerico(self, atributo: str) -> bool:
        return pd.api.types.is_numeric_dtype(self.data[atributo])
    
    def _information_gain(self, atributo: str) -> float:  #IMPORTANTE: este information gain calcula el mejor umbral de ser necesario
        def split(arbol, atributo):
            arbol._split(atributo)
        
        return self.impureza._information_gain_base(self, atributo, split)
    
    def _split_info(self):
        split_info = 0
        len_actual = self._total_samples()
        for subarbol in self.subs:
            len_subarbol = subarbol._total_samples()
            split_info += (len_subarbol / len_actual) * np.log2(len_subarbol / len_actual)
        return -split_info
        
    def _gain_ratio(self, atributo: str) -> float:
        nuevo = self.copy()

        information_gain = nuevo._information_gain(atributo)
        nuevo._split(atributo)
        split_info = nuevo._split_info()

        return information_gain / split_info
    
    def _mejor_atributo_split(self) -> str | None:
        mejor_gain_ratio = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            if len(self.data[atributo].unique()) > 1: # para que no elija columna con un solo valor
                gain_ratio = self._gain_ratio(atributo)
                if gain_ratio > mejor_gain_ratio:
                    mejor_gain_ratio = gain_ratio
                    mejor_atributo = atributo

        return mejor_atributo
    
    def __information_gain_numerico(self, atributo: str, umbral: float | int):  # helper de mejor_umbral_split, no calcula el mejor umbral
            def split_num(arbol, atributo):
                arbol._split_numerico(atributo, umbral)
            
            return self.impureza._information_gain_base(self, atributo, split_num) # clausura, se deberia llevar el umbral
    
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
    
    # TODO: quedo igual al de id3
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target = y.copy()
        self.data = X.copy()
        self.set_clase()

        def _interna(arbol: ArbolClasificadorC45, prof_acum: int = 1):
            arbol.set_target_categorias(y)

            mejor_atributo = arbol._mejor_atributo_split()
            if mejor_atributo and arbol._puede_splitearse(prof_acum, mejor_atributo):
                arbol._split(mejor_atributo) # el check de numerico ahora ocurre dentro de _split()
                
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum + 1)
        
        _interna(self)
        
    def predict(self, X: pd.DataFrame) -> list:
        predicciones = []
        def _recorrer(arbol, fila: pd.Series):
            if arbol.es_hoja():
                return arbol.clase
            else:
                valor = fila[arbol.atributo_split]
                if arbol.es_atributo_numerico(arbol.atributo_split):  # es split numerico
                    if valor < arbol.umbral_split:
                        return _recorrer(arbol.subs[0], fila)
                    else:
                        return _recorrer(arbol.subs[1], fila)
                else:
                    for subarbol in arbol.subs:
                        if valor == subarbol.valor_split_anterior:
                            return _recorrer(subarbol, fila)
                    raise ValueError(f"No se encontró un subárbol para el valor {valor} del atributo {arbol.atributo_split}")
    
        for _, fila in X.iterrows():
            prediccion = _recorrer(self, fila)
            predicciones.append(prediccion)

        return predicciones

    def reduced_error_pruning(self, x_test: Any, y_test: Any):
        def _interna_rep(arbol: ArbolClasificadorC45, x_test, y_test):
            if not arbol.es_hoja():
                for subarbol in arbol.subs:
                    _interna_rep(subarbol, x_test, y_test)

                    pred_raiz: list[str] = arbol.predict(x_test)
                    error_clasif_raiz = Metricas.error(y_test, pred_raiz)

                    error_clasif_ramas = 0.0

                    for rama in arbol.subs:
                        new_arbol: ArbolClasificadorC45 = rama
                        pred_podada = new_arbol.predict(x_test)
                        error_clasif_podada = Metricas.error(y_test, pred_podada)
                        error_clasif_ramas = error_clasif_ramas + error_clasif_podada

                    if error_clasif_ramas < error_clasif_raiz:
                        #print(" * Podar \n")
                        arbol.subs = []
                    #else:
                        #print(" * No podar \n")

        _interna_rep(self, x_test, y_test)

    def __str__(self) -> str:
        out = []
        def _interna(arbol, prefijo: str = '  ', es_ultimo: bool = True) -> None:
            if arbol.signo_split_anterior != "=":
                simbolo_rama = '└─ NO ── ' if es_ultimo else '├─ SI ── '
            else:
                simbolo_rama = '└─── ' if es_ultimo else '├─── '
            
            if arbol.atributo_split and arbol.es_atributo_numerico(arbol.atributo_split):
                split = f"{arbol.atributo_split} < {arbol.umbral_split:.2f} ?" if arbol.umbral_split else ""
            else:
                split = "Split: " + str(arbol.atributo_split)

            
            impureza = f"{arbol.impureza}: {round(arbol.impureza.calcular(arbol.target), 3)}"
            samples = f"Muestras: {arbol._total_samples()}"
            values = f"Conteo: {arbol._values()}"
            clase = f"Clase: {arbol.clase}"

            if arbol.es_raiz():
                out.append(impureza)
                out.append(samples)
                out.append(values)
                out.append(clase)
                out.append(split)

                for i, sub_arbol in enumerate(arbol.subs):
                    ultimo: bool = i == len(arbol.subs) - 1
                    _interna(sub_arbol, prefijo, ultimo)

            elif arbol.es_hoja():
                prefijo_hoja = prefijo + " " * len(simbolo_rama) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                out.append(prefijo + "│")
                            
                if arbol.signo_split_anterior != "=":
                    out.append(prefijo + simbolo_rama + impureza)
                    out.append(prefijo_hoja + samples)
                    out.append(prefijo_hoja + values)
                    out.append(prefijo_hoja + clase)
                else:
                    rta =  f"{arbol.atributo_split_anterior} = {arbol.valor_split_anterior}"
                    out.append(prefijo + simbolo_rama + rta)            
                    out.append(prefijo_hoja + impureza)
                    out.append(prefijo_hoja + samples)
                    out.append(prefijo_hoja + values)
                    out.append(prefijo_hoja + clase)
            
            else:
                out.append(prefijo + "│")
                
                if arbol.atributo_split and arbol.es_atributo_numerico(arbol.atributo_split):
                    out.append(prefijo + simbolo_rama + impureza)
                    prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                    out.append(prefijo2 + samples)
                    out.append(prefijo2 + values)
                    out.append(prefijo2 + clase)
                    out.append(prefijo2 + split)
                else:
                    rta =  f"{arbol.atributo_split_anterior} = {arbol.valor_split_anterior}"
                    out.append(prefijo + simbolo_rama + rta)            
                    prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                    out.append(prefijo2 + impureza)
                    out.append(prefijo2 + samples)
                    out.append(prefijo2 + values)
                    out.append(prefijo2 + clase)
                    out.append(prefijo2 + split)

                prefijo += ' ' * 10 if es_ultimo else '│' + ' ' * 9
                for i, sub_arbol in enumerate(arbol.subs):
                    ultimo: bool = i == len(arbol.subs) - 1
                    _interna(sub_arbol, prefijo, ultimo)
        _interna(self)
        return "\n".join(out)
            
def probar(df, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    arbol = ArbolClasificadorC45(max_prof = 5)
    arbol.fit(x_train, y_train)
    print(arbol)
    arbol.graficar()
    y_pred_train = arbol.predict(x_train)
    y_pred_test = arbol.predict(x_test)
    y_pred_val = arbol.predict(x_val)
    
    print(f"accuracy en set de entrenamiento: {Metricas.accuracy_score(y_train, y_pred_train):.2f}")
    print(f"f1-score en set de entrenamiento: {Metricas.f1_score(y_train, y_pred_train, promedio='ponderado')}\n")

    print(f"accuracy en set de validacion: {Metricas.accuracy_score(y_val, y_pred_val):.2f}")
    print(f"f1-score en set de validacion: {Metricas.f1_score(y_val, y_pred_val, promedio='ponderado')}\n")
    
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred_test):.2f}")
    print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred_test, promedio='ponderado')}\n")
    
    # print("Podo el arbol\n")

    # podado = arbol.reduced_error_pruning2(x_val, y_val)

    # print(podado)
    # podado.graficar()

    # y_pred_train = podado.predict(x_train)
    # y_pred_test = podado.predict(x_test)
    # y_pred_val = podado.predict(x_val)
    
    # print(f"accuracy en set de entrenamiento: {Metricas.accuracy_score(y_train, y_pred_train):.2f}")
    # print(f"f1-score en set de entrenamiento: {Metricas.f1_score(y_train, y_pred_train, promedio='ponderado')}\n")

    # print(f"accuracy en set de validacion: {Metricas.accuracy_score(y_val, y_pred_val):.2f}")
    # print(f"f1-score en set de validacion: {Metricas.f1_score(y_val, y_pred_val, promedio='ponderado')}\n")
    
    # print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred_test):.2f}")
    # print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred_test, promedio='ponderado')}\n")
    

if __name__ == "__main__":
    import sklearn.datasets

    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # print("pruebo con iris")
    # probar(df, "target")

    # print("pruebo con tennis")
    # tennis = pd.read_csv("./datasets/PlayTennis.csv")

    # probar(tennis, "Play Tennis")

    # print("pruebo con patients") 

    # patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    # patients = patients.drop("Patient Id", axis = 1)
    # patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str) # para que sean categorias
    
    # probar(patients, "Level")
    
    titanic = pd.read_csv("./datasets/titanic.csv")
    print("pruebo con titanic")
    probar(titanic, "Survived")