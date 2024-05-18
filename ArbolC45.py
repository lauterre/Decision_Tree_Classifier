# Clase de prueba

from ArbolDecisionID3 import ArbolDecisionID3
import pandas as pd
import numpy as np
from _superclases import ClasificadorArbol, Arbol
from typing import Any, Optional


class ArbolDecisionC45(Arbol, ClasificadorArbol):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1) -> None:
        super().__init__()
        ClasificadorArbol.__init__(self, max_prof, min_obs_nodo)

    def _split(self, atributo: str, valor: Any = None) -> None:
        self.atributo = atributo
        if valor:
            nueva_data_si = self.data[self.data[atributo] <= valor]
            nueva_data_sd = self.data[self.data[atributo] > valor]
            nueva_target_si = self.target[self.data[atributo] <= valor]
            nueva_target_sd = self.target[self.data[atributo] > valor]
            nuevo_si = ArbolDecisionC45()
            nuevo_sd = ArbolDecisionC45()
            nuevo_si.data = nueva_data_si
            nuevo_sd.data = nueva_data_sd
            nuevo_si.target = nueva_target_si
            nuevo_sd.target = nueva_target_sd
            #traer hiper
            # clase
            self.atributo = atributo
            self.valor = valor
            self.subs.append(nuevo_si)
            self.subs.append(nuevo_sd)
        else:
            # si es subclase de id3
            #super()._split(atributo)
            for categoria in self.data[atributo].unique():
                nueva_data = self.data[self.data[atributo] == categoria]
                nueva_data = nueva_data.drop(atributo, axis = 1) # la data del nuevo nodo sin el atributo por el cual ya se filtró
                nuevo_target = self.target[self.data[atributo] == categoria]
                nuevo_arbol = ArbolDecisionID3()
                nuevo_arbol.data = nueva_data
                nuevo_arbol.target = nuevo_target
                nuevo_arbol.valor = categoria
                nuevo_arbol.clase = nuevo_target.value_counts().idxmax()
                nuevo_arbol._traer_hiperparametros(self) # hice un metodo porque van a ser muchos de hiperparametros
                self.subs.append(nuevo_arbol)

        

    def _mejor_umbral_split(self, atributo: str) -> float:

        return float(mejor_umbral)
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target = y
        self.data = X
        self.clase = self.target.value_counts().idxmax()
        
        def _interna(arbol: ArbolDecisionID3, prof_acum: int = 0):
            arbol.target_categorias = y.unique()
            
            if prof_acum == 0:
                prof_acum = 1
            
            if not ( len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                    or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                    or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples() ) ):
                
                mejor_atributo = arbol._mejor_atributo_split()

                if pd.api.types.is_numeric_dtype(self.data[mejor_atributo]): # si es numerica
                    mejor_umbral = arbol._mejor_umbral_split(mejor_atributo)
                arbol._split(mejor_atributo)
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum+1)

        _interna(self)