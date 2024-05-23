from copy import deepcopy
from ArbolDecisionID3 import ArbolDecisionID3
import pandas as pd
import numpy as np
from typing import Any, Optional


class ArbolDecisionC45(ArbolDecisionID3):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.valor_split: Optional[float] = None


    def _nuevo_subarbol(self, atributo: str, operacion: str, valor: Any):
        nuevo = ArbolDecisionC45()
        if operacion == "menor":
            nuevo.data = self.data[self.data[atributo] < valor]
            nuevo.target = self.target[self.data[atributo] < valor]
        elif operacion == "mayor":
            nuevo.data = self.data[self.data[atributo] > valor]
            nuevo.target = self.target[self.data[atributo] > valor]
        elif operacion == "igual":
            nueva_data = self.data[self.data[atributo] == valor]
            nueva_data = nueva_data.drop(atributo, axis = 1)
            nuevo.data = nueva_data
            nuevo.target = self.target[self.data[atributo] == valor]
        nuevo.clase = nuevo.target.value_counts().idxmax()
        nuevo.atributo_split_anterior = atributo
        nuevo.valor_split_anterior = valor
        self.agregar_subarbol(nuevo)

    
    def _split(self, atributo: str, valor: Any = None) -> None:
        self.atributo_split = atributo
        self.valor_split = valor
        if valor:
            self._nuevo_subarbol(atributo, "menor", valor)
            self._nuevo_subarbol(atributo, "mayor", valor)
        else:
            for categoria in self.data[atributo].unique():
                self._nuevo_subarbol(atributo, "igual", categoria)        

    # No me gusta esto de pasar None y tampoco me gusta no sobreescribir los metodos (_information_gain_continuo y demás)
    # preguntar a Mariano
    def _information_gain(self, atributo: str, valor=None) -> float:
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

            information_gain -= ((len_izq/len_actual)*entropia_izq + (len_der/len_actual)*entropia_der)

        else: # si no es continuo
            information_gain =  super()._information_gain(atributo)

        return information_gain   
    
    def _split_info(self):
        split_info = 0
        len_actual = len(self.data)
        for subarbol in self.subs:
            len_subarbol = len(subarbol.data)
            split_info += (len_subarbol/ len_actual) * np.log2(len_subarbol/ len_actual)
        return -split_info
    
    
    def _gain_ratio(self, atributo:str):
        nuevo = deepcopy(self)

        information_gain = nuevo._information_gain(atributo)

        umbral = nuevo._mejor_umbral_split(atributo)
        nuevo._split(atributo, umbral)

        split_info = nuevo._split_info()

        return information_gain / split_info
    
    def _mejor_atributo_split(self) -> str:
        mejor_gain_ratio = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            gain_ratio = self._gain_ratio(atributo)
            if gain_ratio > mejor_gain_ratio:
                mejor_gain_ratio = gain_ratio
                mejor_atributo = atributo
        
        return mejor_atributo
    

    def _mejor_umbral_split(self, atributo: str) -> float:
        self.data = self.data.sort_values(by=atributo)

        mejor_ig = -1
        mejor_umbral = None

        valores_unicos = self.data[atributo].unique()

        i = 0
        while i < len(valores_unicos) - 1:
            umbral = (valores_unicos[i] + valores_unicos[i+1]) / 2
            ig = self._information_gain(atributo, umbral) # uso information_gain, gain_ratio es para la seleccion de atributo
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_umbral = umbral
            i += 1

        return float(mejor_umbral)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target = y
        self.data = X
        self.clase = self.target.value_counts().idxmax()
        
        def _interna(arbol: ArbolDecisionC45, prof_acum: int = 0):
            arbol.target_categorias = y.unique()
            
            if prof_acum == 0:
                prof_acum = 1
            
            if not ( len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                    or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                    or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples() ) ):
                
                mejor_atributo = arbol._mejor_atributo_split()
                
                if pd.api.types.is_numeric_dtype(self.data[mejor_atributo]): # si es numerica
                    mejor_umbral = arbol._mejor_umbral_split(mejor_atributo)
                    arbol._split(mejor_atributo, mejor_umbral)
                else:
                    arbol._split(mejor_atributo)

                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum+1)

        _interna(self)
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
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
            #print(umbral)

            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)

        elif not self.es_hoja():
            print(prefijo + "│")
            #print(prefijo + simbolo_rama + rta)
            print(prefijo + simbolo_rama + entropia)
            prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
            #print(prefijo2 + entropia)
            print(prefijo2 + samples)
            print(prefijo2 + values)
            print(prefijo2 + clase)
            print(prefijo2 + split)
            #print(prefijo2 + umbral)
            
            prefijo += ' '*10 if es_ultimo else '│' + ' '*9
            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)
        else:
            prefijo_hoja = prefijo + " "*len(simbolo_rama) if es_ultimo else prefijo + "│" + " "*(len(simbolo_rama) -1)
            print(prefijo + "│")
            #print(prefijo + simbolo_rama + rta)
            print(prefijo + simbolo_rama + entropia)
            #print(prefijo_hoja + entropia)
            print(prefijo_hoja + samples)
            print(prefijo_hoja + values)
            print(prefijo_hoja + clase)



if __name__ == "__main__":
    import sklearn.datasets
    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    X = df.drop("target", axis = 1)
    y = df["target"]

    arbol = ArbolDecisionC45(max_prof=5, min_obs_nodo = 50)
    arbol.fit(X, y)
    arbol.imprimir()