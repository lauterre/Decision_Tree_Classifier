import numpy as np
import pandas as pd

class Entropia():
    @staticmethod
    def calcular(target: pd.Series) -> float:
        entropia = 0
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia -= proporcion * np.log2(proporcion)
        return round(entropia, 3)

class Gini():
    @staticmethod
    def calcular(target: pd.Series) -> float:
        gini = 1
        proporciones = target.value_counts(normalize=True)
        target_categorias = target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            gini -= proporcion ** 2
        return round(gini, 3)
    