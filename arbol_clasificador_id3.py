import pandas as pd
from herramientas import Herramientas
from metricas import Metricas
from _impureza import Entropia
from _superclases import ArbolClasificador

class ArbolClasificadorID3(ArbolClasificador):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.impureza = Entropia()

    def copy(self):
        nuevo = ArbolClasificadorID3(**self.__dict__) # no funciona bien, solo pasa los hipers?
        nuevo.data = self.data.copy()
        nuevo.target = self.target.copy()
        nuevo.target_categorias = self.target_categorias.copy()
        nuevo.set_clase()
        nuevo.atributo_split = self.atributo_split
        nuevo.atributo_split_anterior = self.atributo_split_anterior
        nuevo.valor_split_anterior = self.valor_split_anterior
        nuevo.signo_split_anterior = self.signo_split_anterior
        nuevo.impureza = self.impureza
        nuevo.subs = [sub.copy() for sub in self.subs]
        return nuevo
    
    def _mejor_atributo_split(self) -> str | None:
        mejor_ig = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            if len(self.data[atributo].unique()) > 1: 
                ig = self._information_gain(atributo)
                if ig > mejor_ig:
                    mejor_ig = ig
                    mejor_atributo = atributo

        return mejor_atributo
        
    def _split(self, atributo: str) -> None:
        self.atributo_split = atributo  # guardo el atributo por el cual spliteo

        for categoria in self.data[atributo].unique():  # recorre el dominio de valores del atributo
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis=1)  # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]

            nuevo_arbol = ArbolClasificadorID3()  # Crea un nuevo arbol
            nuevo_arbol.data = nueva_data  # Asigna nodo
            nuevo_arbol.target = nuevo_target  # Asigna target
            nuevo_arbol.valor_split_anterior = categoria
            nuevo_arbol.atributo_split_anterior = atributo
            nuevo_arbol.set_clase()
            nuevo_arbol.signo_split_anterior = '='
            self.agregar_subarbol(nuevo_arbol)
            
    def _information_gain(self, atributo: str) -> float:
        def split(arbol, atributo):
            arbol._split(atributo)

        return self.impureza.calcular_impureza_split(self, atributo, split) # quizas renombrar a ganancia (o evaluar_split) en impureza     

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
        # TODO: Check no fiteado
        self.target = y.copy()
        self.data = X.copy()
        self.set_clase()

        def _interna(arbol, prof_acum: int = 1):
            arbol.set_target_categorias(y)

            mejor_atributo = arbol._mejor_atributo_split()
            if mejor_atributo and arbol._puede_splitearse(prof_acum, mejor_atributo):
                arbol._split(mejor_atributo)
                
                for sub_arbol in arbol.subs:
                    _interna(sub_arbol, prof_acum + 1)
        
        _interna(self)

    def predict(self, X: pd.DataFrame) -> list[str]:
        predicciones = []
        def _recorrer(arbol, fila: pd.Series) -> None:
            if arbol.es_hoja():
                predicciones.append(arbol.clase)
            else:
                direccion = fila[arbol.atributo_split]
                existe = False
                for subarbol in arbol.subs:
                    if direccion == subarbol.valor_split_anterior:
                        existe = True
                        _recorrer(subarbol, fila)
                if not existe:
                    predicciones.append(predicciones[0])

        for _, fila in X.iterrows():
            _recorrer(self, fila)

        return predicciones
        
    def __str__(self) -> str:
        out = []
        def _interna(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
            simbolo_rama = '└─── ' if es_ultimo else '├─── '
            split = "Split: " + str(self.atributo_split)
            rta =  f"{self.atributo_split_anterior} = {self.valor_split_anterior}"
            impureza = f"{self.impureza}: {round(self.impureza.calcular(self.target), 3)}"
            samples = f"Muestras: {str(self._total_samples())}"
            values = f"Conteo: {str(self._values())}"
            clase = 'Clase: ' + str(self.clase)
            
            if self.es_raiz():
                out.append(impureza)
                out.append(samples)
                out.append(values)
                out.append(clase)
                out.append(split)

                for i, sub_arbol in enumerate(self.subs):
                    ultimo: bool = i == len(self.subs) - 1
                    _interna(sub_arbol, prefijo, ultimo)

            elif not self.es_hoja():
                out.append(prefijo + "│")
                out.append(prefijo + simbolo_rama + rta)
                prefijo2 = prefijo + " " * (len(simbolo_rama)) if es_ultimo else prefijo +"│" + " " * (len(simbolo_rama) - 1)
                out.append(prefijo2 + impureza)
                out.append(prefijo2 + samples)
                out.append(prefijo2 + values)
                out.append(prefijo2 + clase)
                out.append(prefijo2 + split)

                prefijo += ' ' * 10 if es_ultimo else '│' + ' ' * 9
                for i, sub_arbol in enumerate(self.subs):
                    ultimo: bool = i == len(self.subs) - 1
                    _interna(sub_arbol, prefijo, ultimo)
            else:
                prefijo_hoja = prefijo + " " * len(simbolo_rama) if es_ultimo else prefijo + "│" + " " * (len(simbolo_rama) - 1)
                out.append(prefijo + "│")
                out.append(prefijo + simbolo_rama + rta)
                out.append(prefijo_hoja + impureza)
                out.append(prefijo_hoja + samples)
                out.append(prefijo_hoja + values)
                out.append(prefijo_hoja + clase)
        _interna(self)
        return "\n".join(out)

def probar(df, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.2, random_state=42)
    arbol = ArbolClasificadorID3()
    arbol.fit(x_train, y_train)
    print(arbol)
    arbol.graficar()
    y_pred = arbol.predict(x_test)
    
    print(f"\n accuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
    print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")

    # print("Podo el arbol\n")
    # podado = arbol.reduced_error_pruning2(x_test, y_test)
    # print(podado)
    # podado.graficar()
    # y_pred = podado.predict(x_test)    

    # print(f"\n accuracy: {Metricas.accuracy_score(y_test, y_pred):.2f}")
    # print(f"f1-score: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")

if __name__ == "__main__":
    patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)

    tennis = pd.read_csv("./datasets/PlayTennis.csv")

    titanic = pd.read_csv("./datasets/titanic.csv")
    print("Pruebo con Titanic")
    probar(titanic, "Survived")
    # print("Pruebo con patients")
    # probar(patients, "Level")
    # print("Pruebo con Play Tennis")
    # probar(tennis, "Play Tennis")
