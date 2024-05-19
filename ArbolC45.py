from copy import deepcopy
from ArbolDecisionID3 import ArbolDecisionID3
import pandas as pd
import numpy as np
from _superclases import ClasificadorArbol, Arbol
from typing import Any, Optional


class ArbolDecisionC45(ArbolDecisionID3):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1) -> None:
        super().__init__(max_prof, min_obs_nodo)


    def _nuevo_subarbol(self, atributo: str, operacion: str, valor: Any = None):
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
        nuevo.valor = valor
        self.agregar_subarbol(nuevo)

    
    def _split(self, atributo: str, valor: Any = None) -> None:
        self.atributo = atributo
        if valor:
            self._nuevo_subarbol(atributo, "menor", valor)
            self._nuevo_subarbol(atributo, "mayor", valor)
        else:
            for categoria in self.data[atributo].unique():
                self._nuevo_subarbol(atributo, "igual", categoria)        

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
    
    def gain_ratio(self, atributo:str):
        pass
    
    def split_info(self, atributo:str):
        pass

    def _mejor_umbral_split(self, atributo: str) -> float:
        # ordeno la data
        self.data = self.data.sort_values(by=atributo)

        mejor_ig = -1
        mejor_umbral = None

        valores_unicos = self.data[atributo].unique()

        
        # for i, valor in enumerate(valores_unicos):
        #     if i+1 == len(valores_unicos): # feo, pero sirve
        #         break

        #     umbral = (valor + valores_unicos[i+1]) / 2 # el umbral es el valor medio entre valor actual y el siguiente
        #     ig = self._information_gain(atributo, umbral) # uso information_gain, gain_ratio es para la seleccion de atributo
        #     if ig > mejor_ig:
        #         mejor_ig = ig
        #         mejor_umbral = umbral

        i = 0
        while i < len(valores_unicos) - 1:
            umbral = (valores_unicos[i] + valores_unicos[i+1]) / 2 # el umbral es el valor medio entre valor actual y el siguiente
            ig = self._information_gain(atributo, umbral) # uso information_gain, gain_ratio es para la seleccion de atributo
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_umbral = umbral
            i += 1

        return float(mejor_umbral)
        
    # def _gain_ratio(self):
    #     pass

    # sobreescribir para que use gain_ratio
    # def _mejor_atributo_split(self) -> str:
    #     pass


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
                
                mejor_atributo = arbol._mejor_atributo_split() # este metodo usa information_gain y deberia usar gain_ratio, si entendi bien el gain_ratio es solo para eleccion de atributo

                if pd.api.types.is_numeric_dtype(self.data[mejor_atributo]): # si es numerica
                    mejor_umbral = arbol._mejor_umbral_split(mejor_atributo)
                    arbol._split(mejor_atributo, mejor_umbral)
                else:
                    arbol._split(mejor_atributo)

                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum+1)

        _interna(self)
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        simbolo_rama = '└─── ' if es_ultimo else '├─── '
        split = f"Split: {str(self.atributo)}"
        rta = f"Valor: > {str(self.valor)}" if es_ultimo else f"Valor: < {str(self.valor)}"
        entropia = f"Entropia: {round(self._entropia(), 2)}"
        samples = f"Samples: {str (self._total_samples())}"
        values = f"Values: {str(self._values())}"
        clase = 'Clase: ' + str(self.clase)
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
            print(prefijo + simbolo_rama + rta)
            prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
            print(prefijo2 + entropia)
            print(prefijo2 + samples)
            print(prefijo2 + values)
            print(prefijo2 + clase)
            print(prefijo2 + split)
            
            prefijo += ' '*10 if es_ultimo else '│' + ' '*9
            for i, sub_arbol in enumerate(self.subs):
                ultimo: bool = i == len(self.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo)
        else:
            prefijo_hoja = prefijo + " "*len(simbolo_rama) if es_ultimo else prefijo + "│" + " "*(len(simbolo_rama) -1)
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            print(prefijo_hoja + entropia)
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

    arbol = ArbolDecisionC45(max_prof=5)
    arbol.fit(X, y)
    arbol.imprimir()