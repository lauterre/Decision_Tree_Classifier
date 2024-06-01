from copy import deepcopy
from Excepciones import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Metricas import Metricas
from _superclases import ClasificadorArbol, Arbol, Hiperparametros
from Graficador import TreePlot

class ArbolDecisionID3(Arbol, ClasificadorArbol):
    def __init__(self, **kwargs) -> None:
        try:
            super().__init__()
            ClasificadorArbol.__init__(self,**kwargs)
        except Exception as e:
            print(f"Error al inicializar el ArbolDecisionID3: {e}")

    # def agregar_subarbol(self, subarbol):
    #     subarbol.max_prof = self.max_prof
    #     subarbol.min_obs_nodo = self.min_obs_nodo
    #     subarbol.min_infor_gain = self.min_infor_gain
    #     subarbol.min_obs_hoja = self.min_obs_hoja
    #     self.subs.append(subarbol)

        #Cambie este metodo y el de copy para que ande el cambio. 
        #Cumple su funcion de no modificar los init si agregamos hiperparametros, pero el pylance putea cada vez que llamemos un self.hiperparametro (por ejemplo en la interna del fit)

    def agregar_subarbol(self, subarbol):
        try:
            for key, value in self.__dict__.items():
                if key in Hiperparametros().__dict__:  # Solo copiar los atributos que están en Hiperparametros
                    setattr(subarbol, key, value)
            self.subs.append(subarbol)
        except Exception as e:
            print(f"Error al agragr subarbol: {e}")

    # def _traer_hiperparametros(self, arbol_previo):
    #     self.max_prof = arbol_previo.max_prof
    #     self.min_obs_nodo = arbol_previo.min_obs_nodo
    #     self.min_infor_gain = arbol_previo.min_infor_gain
    #     self.min_obs_hoja = arbol_previo.min_obs_hoja

    def _mejor_atributo_split(self) -> str:
        try:
            mejor_ig = -1
            mejor_atributo = ""
            atributos = self.data.columns

            for atributo in atributos:
                ig = self._information_gain(atributo)
                if ig > mejor_ig:
                    mejor_ig = ig
                    mejor_atributo = atributo

            return mejor_atributo
        except Exception as e:
            print(f"Error al tratar de encontrar el mejor atributo de división: {e}")
            return ""

    def copy(self):
        try:
            nuevo = ArbolDecisionID3(**self.__dict__)
            nuevo.data = self.data.copy()
            nuevo.target = self.target.copy()
            nuevo.target_categorias = self.target_categorias.copy()
            nuevo.subs = [sub.copy() for sub in self.subs]
            return nuevo
        except Exception as e:
            print(f"Error al tratar de copiar el arbol: {e}")
            return None

    def _split(self, atributo: str, valor= None) -> None:
        if atributo not in self.data.columns:
            raise AtributoNoEncontradoException(f"Atributo '{atributo}' no encontrado en los datos.")
        temp = deepcopy(self)  # TODO: arreglar copy
        #tmp_subs: list[Arbol]= []
        self.atributo_split = atributo  # guardo el atributo por el cual spliteo

        for categoria in self.data[atributo].unique():  # recorre el dominio de valores del atributo
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis=1)  # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]

            nuevo_arbol = ArbolDecisionID3()  # Crea un nuevo arbol
            nuevo_arbol.data = nueva_data  # Asigna nodo
            nuevo_arbol.target = nuevo_target  # Asigna target
            nuevo_arbol.valor_split_anterior = categoria
            nuevo_arbol.atributo_split_anterior = atributo
            nuevo_arbol.clase = nuevo_target.value_counts().idxmax()
            temp.agregar_subarbol(nuevo_arbol)  # Agrego el nuevo arbol en el arbol temporal

        ok_min_obs_hoja = True
        for sub_arbol in temp.subs:
            if (self.min_obs_hoja !=-1 and sub_arbol._total_samples() < self.min_obs_hoja):
                ok_min_obs_hoja = False

        if ok_min_obs_hoja:
            self.subs = temp.subs
        

    def _entropia(self) -> float:
        try:
            entropia = 0
            proporciones = self.target.value_counts(normalize=True)
            target_categorias = self.target.unique()
            for c in target_categorias:
                proporcion = proporciones.get(c, 0)
                entropia += proporcion * np.log2(proporcion)
            return -entropia if entropia != 0 else 0
        except Exception as e:
            print(f"Error al tratar de calcular la entropia: {e}")
            return 0

    def _information_gain(self, atributo: str, valor = None) -> float:
        try:
            entropia_actual = self._entropia()
            len_actual = len(self.data)

            nuevo = self.copy()
            if nuevo is None:
                raise ValorInvalidoException("Fallo al tratar de copiar el árbol")

            nuevo._split(atributo)

            entropias_subarboles = 0 
            for subarbol in nuevo.subs:
                entropia = subarbol._entropia()
                len_subarbol = len(subarbol.data)
                entropias_subarboles += ((len_subarbol/len_actual) * entropia)

            information_gain = entropia_actual - entropias_subarboles
            return information_gain
        except Exception as e:
            print(f"Error al tratar de calcular el information gain: {e}")
            return 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame):
            raise ConversionDeTiposException("X debe ser un DataFrame de pandas")
        if not isinstance(y, pd.Series):
            raise ConversionDeTiposException("y debe ser una Serie de pandas")
        try:
            self.target = y
            self.data = X
            self.clase = self.target.value_counts().idxmax()

            def _interna(arbol: ArbolDecisionID3, prof_acum: int = 0):
                try:
                    arbol.target_categorias = y.unique()

                    if prof_acum == 0:
                        prof_acum = 1

                    if not (len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                            or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                            or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples())):

                        mejor_atributo = arbol._mejor_atributo_split()
                        mejor_ig = arbol._information_gain(mejor_atributo)

                        if (arbol.min_infor_gain == -1 or mejor_ig >= arbol.min_infor_gain):
                            arbol._split(mejor_atributo)

                            for sub_arbol in arbol.subs:
                                _interna(sub_arbol, prof_acum + 1)
                except Exception as e:
                    print(f"Error en la funcion interna del fit: {e}")

            _interna(self)
        except Exception as e:
            print(f"Error al tratar de fittear el arbol: {e}")

    def predict(self, X: pd.DataFrame) -> list[str]:
        if X.empty:
            raise ValorInvalidoException("El conjunto de datos de entrada está vacío")
        else:
            predicciones = []
            def _recorrer(arbol, fila: pd.Series) -> None:
                if arbol.es_hoja():
                    predicciones.append(arbol.clase)
                else:
                    direccion = fila[arbol.atributo_split]
                    existe = False
                    for subarbol in arbol.subs:
                        if direccion == subarbol.valor_split_anterior:  # subarbol.valor
                            existe = True
                            _recorrer(subarbol, fila)
                    if not existe:
                        predicciones.append(predicciones[0])

            for _, fila in X.iterrows():
                _recorrer(self, fila)

    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        try:
            simbolo_rama = '└─── ' if es_ultimo else '├─── '
            split = "Split: " + str(self.atributo_split)
            rta =  f"{self.atributo_split_anterior} = {self.valor_split_anterior}"
            entropia = f"Entropia: {round(self._entropia(), 2)}"
            samples = f"Samples: {str(self._total_samples())}"
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

                prefijo += ' ' * 10 if es_ultimo else '│' + ' ' * 9
                for i, sub_arbol in enumerate(self.subs):
                    ultimo: bool = i == len(self.subs) - 1
                    sub_arbol.imprimir(prefijo, ultimo)
            else:
                prefijo_hoja = prefijo + " " * len(simbolo_rama) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                print(prefijo + "│")
                print(prefijo + simbolo_rama + rta)
                print(prefijo_hoja + entropia)
                print(prefijo_hoja + samples)
                print(prefijo_hoja + values)
                print(prefijo_hoja + clase)
        except Exception as e:
            print(f"Error al tratar de imprimir el arbol: {e}")

    def graficar(self):
        try:
            plotter = TreePlot(self)
            plotter.plot()
        except Exception as e:
            print(f"Error al tratar de graficar el arbol: {e}")

    def _error_clasificacion(self, y, y_pred):
        if len(y) != len(y_pred):
            raise LongitudInvalidaException("Las longitudes de y y y_pred no coinciden")
        else:
            x = []
            for i in range(len(y)):
                x.append(y[i] != y_pred[i])
            return np.mean(x)

    def Reduced_Error_Pruning(self, x_test: any, y_test: any):
        try:
            def _interna_REP(arbol: ArbolDecisionID3, x_test, y_test):
                if arbol.es_hoja():
                    return

                for subarbol in arbol.subs:
                    _interna_REP(subarbol, x_test, y_test)

                try:
                    pred_raiz: list[str] = arbol.predict(x_test)
                    accuracy_raiz = Metricas.accuracy_score(y_test.tolist(), pred_raiz)
                    error_clasif_raiz = arbol._error_clasificacion(y_test.tolist(), pred_raiz)

                    error_clasif_ramas = 0.0

                    for rama in arbol.subs:
                        new_arbol: ArbolDecisionID3 = rama
                        pred_podada = new_arbol.predict(x_test)
                        accuracy_podada = Metricas.accuracy_score(y_test.tolist(), pred_podada)
                        error_clasif_podada = new_arbol._error_clasificacion(y_test.tolist(), pred_podada)
                        error_clasif_ramas = error_clasif_ramas + error_clasif_podada

                    if error_clasif_ramas < error_clasif_raiz:
                        print(" * Podar \n")
                        arbol.subs = []
                    else:
                        print(" * No podar \n")
                except Exception as e:
                    print(f"Error en la funcion _interna_REP: {e}")

            _interna_REP(self, x_test, y_test)
        except Exception as e:
            print(f"Error en la función Reduced Error Pruning: {e}")

                # if precision_podada > mejor_precision:
            #     mejor_rama = rama
            #     mejor_precision = precision_podada

            # if mejor_rama is not None:
            #     arbol_podado = podar_rama(arbol, mejor_rama)
            #     return REP(arbol_podado, conjunto_validacion)
            # else:
            #     return arbol
    
            # for subarbol in arbol.subs: 

def probar(df, target: str):
        X = df.drop(target, axis=1)
        y = df[target]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #arbol = ArbolDecisionID3(min_obs_nodo=1)
        #arbol = ArbolDecisionID3(min_infor_gain=0.85)
        arbol = ArbolDecisionID3(min_obs_nodo=1)
        arbol.fit(x_train, y_train)
        arbol.imprimir()
        y_pred = arbol.predict(x_test)

        arbol.Reduced_Error_Pruning(x_test, y_test)

        print(f"\naccuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
        
        print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")
        

if __name__ == "__main__":
    #https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
    patients = pd.read_csv("cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)

    tennis = pd.read_csv("PlayTennis.csv")

    print("Pruebo con patients")
    probar(patients, "Level")
    print("Pruebo con Play Tennis")
    probar(tennis, "Play Tennis")
