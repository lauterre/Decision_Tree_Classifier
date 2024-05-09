from typing import Optional
from functools import wraps
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from copy import deepcopy

class Nodo:
    def __init__(self, data: pd.DataFrame, target: pd.Series) -> None: #data : data de entrenamiento
        self.atributo: Optional[str] = None # se setea cuando se haga un split
        self.categoria: Optional[str] = None # LA CATEGORIA RESULTANTE DEL SPLIT ANTERIOR
        self.data: pd.DataFrame = data
        self.target: pd.Series = target
        self.clase: Optional[str] = None # TODO: actualmente se setea cuando sea hoja, pero siempre deberia tener un valor, por ej: el valor que más se repite
        self.subs: list[ArbolID3] = []

    def _split_id3(self, atributo: str) -> None:
        self.atributo = atributo # guardo el atributo por el cual spliteo
        for categoria in self.data[atributo].unique():
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis = 1) # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]
            nuevo = Nodo(nueva_data, nuevo_target)
            nuevo.categoria = categoria
            self.subs.append(ArbolID3("ID3", nuevo))
            
    
    def entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize= True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia
    
class ArbolID3:
    def __init__(self,type: str, nodo: Optional[Nodo] = None) -> None:  
        #TODO: Ver como inicializamos el arbol. El Optional rompe todas las typehints, y estaria bueno no tenerlo.
        #      Lo resolvi de forma que la data la pasamos en el fit directamente
        self.type: str = type
        self.raiz = nodo

    @staticmethod
    def crear_arbol(type: str = "ID3"):
        return ArbolID3(type)
    
    def _mejor_split(self) -> str:
        mejor_ig = -1
        atributos = self.raiz.data.columns

        for atributo in atributos:
            # for categoria in self.raiz.data[atributo].unique():
            ig = self._information_gain(atributo)
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_atributo = atributo
        
        return mejor_atributo

    def fit(self,df: pd.DataFrame, target: pd.Series) -> None:
        self.raiz = Nodo(df, target)
        if self.type == "ID3":
            self._fit_id3()
    
    def _fit_id3(self) -> None:
        if len(self.raiz.target.unique()) == 1 or len(self.raiz.data.columns) == 0:
            self.raiz.clase = self.raiz.target.value_counts().idxmax()
        else:
            mejor_atributo = self._mejor_split()
            self.raiz._split_id3(mejor_atributo)
            [sub_arbol._fit_id3() for sub_arbol in self.raiz.subs]
            
        
    def _information_gain(self, atributo: str) -> float:
        entropia_actual = self.raiz.entropia()
        len_actual = len(self.raiz.data)

        nuevo = deepcopy(self)
        nuevo.raiz._split_id3(atributo)

        entropias_subarboles = 0 # no son las entropias, son |D_v|/|D| * Entropia(D_v)

        for subarbol in nuevo.raiz.subs:
            entropia = subarbol.raiz.entropia()
            len_subarbol = len(subarbol.raiz.data)
            entropias_subarboles += ((len_subarbol/len_actual)*entropia)

        information_gain = entropia_actual - entropias_subarboles
        return information_gain
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True, es_raiz: bool = True) -> None:
        nodo = self.raiz
        simbolo_rama = '└─── ' if es_ultimo else '├─── '
        pregunta = str(nodo.atributo) + "?"
        #len_str = len(str(nodo.data[nodo.atributo].max()))
        rta = "Valor: " + str(nodo.categoria)
        entropia = f"Entropia: {round(self.raiz.entropia(), 2)}"
        samples = f"Samples: {len(self.raiz.data)}"
        if es_raiz:
            print(entropia)
            print(samples)
            print(pregunta)

            for i, sub_arbol in enumerate(nodo.subs):
                ultimo: bool = i == len(nodo.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo, False)

        elif nodo.atributo is not None:
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
            print(prefijo2 + entropia)
            print(prefijo2 + samples)
            print(prefijo2 + pregunta)
            
            prefijo += ' '*10 if es_ultimo else '│' + ' '*9
            for i, sub_arbol in enumerate(nodo.subs):
                ultimo: bool = i == len(nodo.subs) - 1
                sub_arbol.imprimir(prefijo, ultimo, False)
        else:
            prefijo_hoja = prefijo + " "*len(simbolo_rama) if es_ultimo else prefijo + "│" + " "*(len(simbolo_rama) -1)
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            print(prefijo_hoja + entropia)
            print(prefijo_hoja + samples)
            print(prefijo_hoja + 'Clase:', str(nodo.clase))

    def predict(self, X: pd.DataFrame) -> list[str]:
        predicciones = []

        def _interna(arbol, X):
            nodo = arbol.raiz
            if not nodo.subs:  # es hoja
                predicciones.append(nodo.clase)
            else:
                atributo = nodo.atributo
                # categoria = nodo.categoria
                valor_atributo = X[atributo].iloc[0]
                for i, subarbol in enumerate(nodo.subs):
                    if valor_atributo == subarbol.raiz.categoria:
                        _interna(arbol.raiz.subs[i], X)

        for _, row in X.iterrows():
            _interna(self, pd.DataFrame([row]))
        
        return predicciones


if __name__ == "__main__":
    #https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
    df = pd.read_csv("G:/algo2/TP_Final_algo2/TP_Final/cancer_patients.csv", index_col=0)
    df = df.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    X = df.drop('Level', axis=1)
    y = df['Level']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    arbol = ArbolID3.crear_arbol()

    arbol.fit(x_train, y_train) # acá deberian ir x_train e y_train, no en crear_arbol

    arbol.imprimir()

    y_pred = arbol.predict(x_test)

    def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError()
        correctas = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        precision = correctas / len(y_true)
        return precision
    
    print(accuracy_score(y_test.tolist(), y_pred))
