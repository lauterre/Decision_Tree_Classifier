import pandas as pd
import numpy as np
from arbol_clasificador_c45 import ArbolClasificadorC45
from metricas import Metricas
from herramientas import Herramientas, GridSearch
from _superclases import Clasificador, Bosque, Hiperparametros
from arbol_clasificador_id3 import ArbolClasificadorID3


class BosqueClasificador(Bosque, Clasificador):
    def __init__(self, clase_arbol: str = "id3", cantidad_arboles: int = 10, cantidad_atributos:str ='sqrt',verbose: bool = False,**kwargs) -> None:
        super().__init__(cantidad_arboles)
        self.hiperparametros_arbol = Hiperparametros(**kwargs)
        for key, value in self.hiperparametros_arbol.__dict__.items():
            setattr(self, key, value)
        self.cantidad_atributos = cantidad_atributos
        self.clase_arbol = clase_arbol
        self.verbose = verbose

    def _bootstrap_samples(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        # Nro filas
        n_samples = X.shape[0]
        atributos = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[atributos].reset_index(drop=True), y.iloc[atributos].reset_index(drop=True)

    def seleccionar_atributos(self, X: pd.DataFrame)-> list[int]:
        n_features = X.shape[1]
        if self.cantidad_atributos == 'all':
            size = n_features
        elif self.cantidad_atributos == 'log2':
            size = int(np.log2(n_features))
        elif self.cantidad_atributos == 'sqrt':
            size = int(np.sqrt(n_features))
        else:
            pass
            #TODO: agregar exception

        indices = np.random.choice(n_features, size, replace=False)
        return indices

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for _ in range(self.cantidad_arboles):
            if self.verbose : print(f"Contruyendo arbol nro: {_ + 1}") 
            # Bootstrapping
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Selección de atributos
            atributos = self.seleccionar_atributos(X_sample)
            X_sample = X_sample.iloc[:, atributos]

            # Crear y entrenar un nuevo árbol
            if self.clase_arbol == 'id3':
                arbol = ArbolClasificadorID3(**self.hiperparametros_arbol.__dict__)
                arbol.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                self.arboles.append(arbol)
            elif self.clase_arbol == 'c45':
                arbol = ArbolClasificadorC45(**self.hiperparametros_arbol.__dict__)
                arbol.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                self.arboles.append(arbol)
            else:
                raise ValueError("Clase de arbol soportado por el bosque: 'id3', 'c45'")
            #arbol.imprimir()

    def predict(self, X: pd.DataFrame) -> list:
        todas_predicciones = pd.DataFrame(index=X.index, columns=range(len(self.arboles))) 
        
        for i, arbol in enumerate(self.arboles):
            todas_predicciones[i] = arbol.predict(X)

        # Aplicar la votación mayoritaria
        predicciones_finales = todas_predicciones.apply(lambda x: x.value_counts().idxmax(), axis=1)
        
        return list(predicciones_finales)
    
def probar_bosque_clasificador(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    rf = BosqueClasificador(clase_arbol="id3", cantidad_arboles = 10, cantidad_atributos='sqrt', max_prof=2, min_obs_nodo=10, verbose=True)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred)}")
    print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")
    

def probar_cv(df: pd.DataFrame, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.15, random_state=42)
    rf = BosqueClasificador(clase_arbol="c45", cantidad_arboles = 10, cantidad_atributos='sqrt', max_prof=2, min_obs_nodo=10, verbose=True)
    print(Herramientas.cross_validation(x_train, y_train, rf, 5,verbose=True))

def probar_grid_search(df, target: str):
    X = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    rf = BosqueClasificador()
    grid_search = GridSearch(rf, {'clase_arbol': ['id3', 'c45'],'min_obs_nodo': [10, 20, 30, 40]}, k_fold=3)

    grid_search.fit(x_train, y_train)
    print(grid_search.mejores_params)
    print(grid_search.mejor_score)
    print(grid_search.mostrar_resultados())
    mejor_bosque = BosqueClasificador(**grid_search.mejores_params)
    mejor_bosque.fit(x_train, y_train)

    y_pred = mejor_bosque.predict(x_test)
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred)}")
    print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")
    
    
if __name__ == "__main__":
    # print("pruebo con patients") 

    # patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    # patients = patients.drop("Patient Id", axis = 1)
    # patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str) # para que sean categorias
    # print("pruebo bosque clasificador")
    # #probar_bosque_clasificador(patients, "Level") #anda joya
    
    # print("pruebo cross validation")
    # #probar_cv(patients, "Level") #anda joya
    # print("pruebo grid search")
    # probar_grid_search(patients, "Level") #anda joya, ojo con correrlo que tarda bastante (hay muchas combinaciones)

    print("pruebo con titanic")
    titanic = pd.read_csv("./datasets/titanic.csv")
    probar_cv(titanic, "Survived")


        