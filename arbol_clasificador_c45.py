import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import random
from typing import Any, Optional

from _impureza import Entropia
from _superclases import ArbolClasificador
from herramientas import GridSearch, Herramientas
from metricas import Metricas


class ArbolClasificadorC45(ArbolClasificador):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.umbral_split: Optional[float | str] = None
        self.impureza = Entropia()
    
    def copy(self):
        nuevo = ArbolClasificadorC45(**self.__dict__) 
        nuevo.data = self.data.copy()
        nuevo.target = self.target.copy()
        nuevo.target_categorias = self.target_categorias.copy()
        nuevo.set_clase()
        nuevo.atributo_split = self.atributo_split
        nuevo.atributo_split_anterior = self.atributo_split_anterior
        nuevo.valor_split_anterior = self.valor_split_anterior
        nuevo.signo_split_anterior = self.signo_split_anterior
        nuevo.impureza = self.impureza
        nuevo.umbral_split = self.umbral_split 
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
    
    def _split_ordinal(self, atributo: str, umbral: str) -> None:
        self.atributo_split = atributo
        self.umbral_split = umbral
        self._nuevo_subarbol(atributo, "menor", umbral)
        self._nuevo_subarbol(atributo, "mayor", umbral)
    
    def _split(self, atributo: str):
        if self.es_atributo_numerico(atributo):
            mejor_umbral = self._mejor_umbral_split_num(atributo)
            self._split_numerico(atributo, mejor_umbral)

        elif self.es_atributo_ordinal(atributo):
            mejor_umbral = self._mejor_umbral_split_ord(atributo)
            self._split_ordinal(atributo, mejor_umbral)

        else:
            self._split_categorico(atributo)

    def es_atributo_numerico(self, atributo: str) -> bool:
        return pd.api.types.is_numeric_dtype(self.data[atributo])
    
    def es_atributo_ordinal(self, atributo: str) -> bool:
        return isinstance(self.data[atributo].dtype, pd.CategoricalDtype) and self.data[atributo].cat.ordered
    
    def _information_gain(self, atributo: str) -> float:  #calcula el mejor umbral de ser necesario
        def split(arbol, atributo):
            arbol._split(atributo)
        
        return self.impureza.calcular_impureza_split(self, atributo, split)
    
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
            if len(self.data[atributo].unique()) > 1:
                gain_ratio = self._gain_ratio(atributo)
                if gain_ratio > mejor_gain_ratio:
                    mejor_gain_ratio = gain_ratio
                    mejor_atributo = atributo

        return mejor_atributo
    
    def __information_gain_numerico(self, atributo: str, umbral: float | int):  # helper de mejor_umbral_split, no calcula el mejor umbral
            def split_num(arbol, atributo):
                arbol._split_numerico(atributo, umbral)
            
            return self.impureza.calcular_impureza_split(self, atributo, split_num)
    
    def __information_gain_ordinal(self, atributo: str, umbral: str):  # helper de mejor_umbral_split, no calcula el mejor umbral
            def split_ord(arbol, atributo):
                arbol._split_ordinal(atributo, umbral)

            return self.impureza.calcular_impureza_split(self, atributo, split_ord)
    
    def _mejor_umbral_split_num(self, atributo: str) -> float:
        self.data = self.data.sort_values(by=atributo)
        mejor_ig = -1
        valores_unicos = self.data[atributo].unique()
        mejor_umbral = valores_unicos[0]
        
        i = 0
        while i < len(valores_unicos) - 1:
            umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
            ig = self.__information_gain_numerico(atributo, umbral)
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_umbral = umbral
            i += 1
        
        return mejor_umbral  
    
    def _mejor_umbral_split_ord(self, atributo: str) -> str:
        self.data = self.data.sort_values(by=atributo, ascending=False) # porque cuando hacemos el split pedimos que sea < al umbral, si estan de menor a mayor, el primer valor nunca es < al umbral
        mejor_ig = -1
        valores_unicos = self.data[atributo].unique()
        mejor_umbral = valores_unicos[0]

        i = 0
        while i < len(valores_unicos) - 1:
            ig = self.__information_gain_ordinal(atributo, valores_unicos[i])
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_umbral = valores_unicos[i]
            i += 1
        
        return mejor_umbral
    
    def _rellenar_missing_values(self):
        for columna in self.data.columns:
            if self.es_atributo_numerico(columna):
                medias = self.data.groupby(self.target)[columna].transform('mean')
                self.data.fillna({columna: medias}, inplace=True)
            else:
                modas = self.data.groupby(self.target)[columna].transform(lambda x: x.mode()[0] if not x.mode().empty else x)
                self.data.fillna({columna: modas}, inplace=True)
    
    def _puede_splitearse(self, prof_acum: int, mejor_atributo: str) -> bool:
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO: check no fitteado
        self.target = y.copy()
        self.data = X.copy()
        self._rellenar_missing_values()
        self.set_clase()
        
        def _interna(arbol, prof_acum: int = 1):
            arbol.set_target_categorias(y)

            mejor_atributo = arbol._mejor_atributo_split()
            if mejor_atributo and arbol._puede_splitearse(prof_acum, mejor_atributo):
                arbol._split(mejor_atributo)
                
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
                if pd.isna(valor):  # Manejar valores faltantes en la predicción
                    dist_probabilidades = _predict_valor_faltante(arbol, fila)
                    return obtener_clase_aleatoria(dist_probabilidades)          
                if arbol.es_atributo_numerico(arbol.atributo_split):  # es split numerico, TODO: lo podes ver con el signo
                    if valor < arbol.umbral_split:
                        return _recorrer(arbol.subs[0], fila)
                    else:
                        return _recorrer(arbol.subs[1], fila)
                elif arbol.es_atributo_ordinal(arbol.atributo_split):
                    niveles_ordenados = arbol.data[arbol.atributo_split].cat.categories
                    posicion_valor = niveles_ordenados.get_loc(valor)
                    posicion_umbral = niveles_ordenados.get_loc(arbol.umbral_split)
                    if posicion_valor < posicion_umbral:
                        return _recorrer(arbol.subs[0], fila)
                    else:
                        return _recorrer(arbol.subs[1], fila)
                else:
                    for subarbol in arbol.subs:
                        if valor == subarbol.valor_split_anterior:
                            return _recorrer(subarbol, fila)
                    #raise ValueError(f"No se encontró un subárbol para el valor {valor} del atributo {arbol.atributo_split}")

        def _predict_valor_faltante(arbol, fila):
            total_samples = arbol._total_samples()
            probabilidades = {clase: 0 for clase in arbol.target_categorias}
            for subarbol in arbol.subs:
                sub_samples = subarbol._total_samples()
                sub_prob = sub_samples / total_samples
                if subarbol.es_hoja():
                    for i, clase in enumerate(arbol.target_categorias):
                        probabilidades[clase] += subarbol._values()[i] 
                else:
                    sub_probs = _predict_valor_faltante(subarbol, fila) 
                    for i, clase in enumerate(arbol.target_categorias):
                        probabilidades[clase] += sub_probs[clase] 
            return probabilidades
        
        def obtener_clase_aleatoria(diccionario_probabilidades):
                cantidad_total = sum(diccionario_probabilidades.values())
                for key in diccionario_probabilidades:
                    diccionario_probabilidades[key] = round(diccionario_probabilidades[key] / cantidad_total,2) #Redondeado para que se entienda mas
                total_valores = sum(diccionario_probabilidades.values())
                probabilidades = [valor / total_valores for valor in diccionario_probabilidades.values()]
                clases = list(diccionario_probabilidades.keys())
                
                clase_aleatoria = random.choices(clases, weights=probabilidades, k=1)[0]
                print(diccionario_probabilidades,clase_aleatoria)
                return clase_aleatoria

        for _, fila in X.iterrows():
            prediccion = _recorrer(self, fila)
            #print(prediccion)
            predicciones.append(prediccion)

        return predicciones

    def __str__(self) -> str:
        out = []
        def _interna(arbol, prefijo: str = '  ', es_ultimo: bool = True) -> None:
            if arbol.signo_split_anterior != "=":
                simbolo_rama = '└─ NO ── ' if es_ultimo else '├─ SI ── '
            else:
                simbolo_rama = '└─── ' if es_ultimo else '├─── '
            
            if arbol.atributo_split and arbol.es_atributo_numerico(arbol.atributo_split):
                split = f"{arbol.atributo_split} < {round(arbol.umbral_split, 2)} ?" if arbol.umbral_split else ""
            elif arbol.atributo_split and arbol.es_atributo_ordinal(arbol.atributo_split):
                split = f"{arbol.atributo_split} = {arbol.umbral_split} ?" if arbol.umbral_split else ""
            else:
                split = str(arbol.atributo_split) + " ?"

            
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

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    x_train, x_val, x_test, y_train, y_val, y_test = Herramientas.dividir_set(X, y, test_size=0.15, val_size=0.15, val=True, random_state=42)
    arbol = ArbolClasificadorC45()

    arbol.fit(X, y)
    print(arbol)
    arbol.graficar()
    y_pred_train = arbol.predict(x_train)
    
    
    print(f"accuracy en set de entrenamientosadasdadsad: {Metricas.accuracy_score(y_train, y_pred_train)}")
    print(f"f1-score en set de entrenamientoasdadasdasd: {Metricas.f1_score(y_train, y_pred_train, promedio='ponderado')}\n")

    # print(f"accuracy en set de validacion: {Metricas.accuracy_score(y_val, y_pred_val)}")
    # print(f"f1-score en set de validacion: {Metricas.f1_score(y_val, y_pred_val, promedio='ponderado')}\n")
    
    # print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred_test)}")
    # print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred_test, promedio='ponderado')}\n")
    
    # print("Podo el arbol\n")

    # arbol.reduced_error_pruning(x_val, y_val)

    # #print(podado)
    # arbol.graficar()

    # y_pred_train = arbol.predict(x_train)
    # y_pred_test = arbol.predict(x_test)
    # y_pred_val = arbol.predict(x_val)
    
    # print(f"accuracy en set de entrenamiento: {Metricas.accuracy_score(y_train, y_pred_train):.2f}")
    # print(f"f1-score en set de entrenamiento: {Metricas.f1_score(y_train, y_pred_train, promedio='ponderado')}\n")

    # print(f"accuracy en set de validacion: {Metricas.accuracy_score(y_val, y_pred_val)}")
    # print(f"f1-score en set de validacion: {Metricas.f1_score(y_val, y_pred_val, promedio='ponderado')}\n")
    
    # print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred_test)}")
    # print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred_test, promedio='ponderado')}\n")

def probar_cv(df, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.15, random_state=42)
    arbol = ArbolClasificadorC45(max_prof = 5, min_obs_hoja=5, min_obs_nodo=5)
    print(Herramientas.cross_validation(x_train, y_train, arbol, 5, verbose=True))

def probar_grid_search(df, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    arbol = ArbolClasificadorC45()
    grid_search = GridSearch(arbol, {"max_prof": [2, 3, -1], "min_obs_hoja": [3, 5,-1], "min_obs_nodo": [3, 5,-1]}, k_fold=3)

    grid_search.fit(x_train, y_train)
    print(grid_search.mejores_params)
    print(grid_search.mejor_score)
    print(grid_search.mostrar_resultados())
    mejor_arbol = ArbolClasificadorC45(**grid_search.mejores_params)
    mejor_arbol.fit(x_train, y_train)
    mejor_arbol.graficar()

    y_pred = mejor_arbol.predict(x_test)
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred)}")
    print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")

if __name__ == "__main__":
    import sklearn.datasets

    # iris = sklearn.datasets.load_iris()
    # df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # df['target'] = iris.target

<<<<<<< Updated upstream
    # print("pruebo con iris")
    # probar(df, "target")
    # probar_cv(df, "target")
    #probar_grid_search(df, "target")
=======
    print("pruebo con iris")
    probar(df, "target")
    probar_cv(df, "target")
    probar_grid_search(df, "target")
>>>>>>> Stashed changes

    # print("pruebo con tennis")
    # tennis = pd.read_csv("./datasets/PlayTennis.csv")
    # temperature_order = ['Cool', 'Mild', 'Hot']
    # humidity_order = ['Normal', 'High']
    # wind_order = ['Weak', 'Strong']

    # Convertir columnsas a ordinal
    # tennis['Temperature'] = tennis['Temperature'].astype(CategoricalDtype(categories=temperature_order, ordered=True))
    # tennis['Humidity'] = tennis['Humidity'].astype(CategoricalDtype(categories=humidity_order, ordered=True))
    # tennis['Wind'] = tennis['Wind'].astype(CategoricalDtype(categories=wind_order, ordered=True))
    #probar_grid_search(tennis, "Play Tennis")
    # probar_cv(df, "Play Tennis")
    # probar(tennis, "Play Tennis")

    # print("pruebo con patients") 

    # patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    # patients = patients.drop("Patient Id", axis = 1)
    # patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str) # para que sean categorias
    
    # #probar_cv(patients, "Level")
    # probar_grid_search(patients, "Level")

    # print("pruebo con patientsna")
    # patientsna = pd.read_csv("./datasets/cancer_patients_con_NA.csv", index_col=0)
    # patientsna = patientsna.drop("Patient Id", axis = 1)
    # patientsna.loc[:, patientsna.columns != "Age"] = patientsna.loc[:, patientsna.columns != "Age"].astype(str) # para que sean categorias
    # probar_grid_search(patientsna, "Level")
    
    titanic = pd.read_csv("./datasets/titanic.csv")
    print("pruebo con titanic")
    probar_cv(titanic, "Survived")
    #probar(titanic, "Survived")
    #probar_grid_search(titanic, "Survived")
