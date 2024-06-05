from copy import deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Metricas import Metricas
from _superclases import ClasificadorBosque, Bosque, Hiperparametros, Arbol
from ArbolDecisionID3 import ArbolDecisionID3

class RandomForest(Bosque, ClasificadorBosque):
    def __init__(self,clase_arbol: str = "id3", cantidad_arboles: int = 10, cantidad_atributos:str ='sqrt',**kwargs):
        super().__init__(clase_arbol, cantidad_arboles, cantidad_atributos)
        ClasificadorBosque.__init__(self, **kwargs)

    def seleccionar_atributos(self, X: pd.DataFrame)-> list[int]:
        n_features = X.shape[1]
        if self.cantidad_atributos == 'sqrt':
            size = int(np.sqrt(n_features))
        elif self.cantidad_atributos == 'log2':
            size = int(np.log2(n_features))
        else:
            size = n_features

        indices = np.random.choice(n_features, size, replace=False)
        return indices


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for _ in range(self.cantidad_arboles):
            print(f"Contruyendo arbol nro: {_ + 1}")
            # Bootstrapping
            X_sample, y_sample = self._bootstrap_samples(X, y)

            # Selección de atributos
            atributos = self.seleccionar_atributos(X_sample)
            X_sample = X_sample.iloc[:, atributos]

            # Crear y entrenar un nuevo árbol
            if self.clase_arbol == 'id3':
                arbol = ArbolDecisionID3(max_prof=self.max_prof, min_obs_nodo=self.min_obs_nodo)
                arbol.fit(pd.DataFrame(X_sample), pd.Series(y_sample))
                self.arboles.append(arbol)
            else:
                raise ValueError("Clase de arbol soportado por el bosque: 'id3'")
            #arbol.imprimir()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        todas_predicciones = pd.DataFrame(index=X.index, columns=range(len(self.arboles))) 
        
        for i, arbol in enumerate(self.arboles):
            todas_predicciones[i] = arbol.predict(X)

        # Aplicar la votación mayoritaria
        predicciones_finales = todas_predicciones.apply(lambda x: x.value_counts().idxmax(), axis=1)
        
        return predicciones_finales
    
    def cross_validate(features, target, classifier, k_fold) :

        # derive a set of (random) training and testing indices
        k_fold_indices = KFold(len(features), n_folds=k_fold,
                            shuffle=True, random_state=0)
        # for each training and testing slices run the classifier, and score the results
        k_score_total = 0
        for train_slice, test_slice in k_fold_indices :
            model = classifier.fit(features[train_slice],
                                target[train_slice])
            k_score = model.score(features[test_slice],
                                target[test_slice])
            k_score_total += k_score
        # return the average accuracy
        return k_score_total/k_fold

if __name__ == "__main__":
    # Crea un conjunto de datos de ejemplo
    patients = pd.read_csv("cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)
    X = patients.drop('Level', axis=1)
    y = patients["Level"]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    lista_indices = x_train.index
    folds = 3
    group_size = len(lista_indices) // folds
    groups = []
    for i in range (folds):
        desde = i * group_size
        hasta = desde + group_size
        groups.append(lista_indices[i:hasta])
    
    if len(groups[-1]) < group_size:  # Handle the case when the last group has fewer numbers
        remaining_numbers: int = groups.pop()
        groups.append(remaining_numbers[:group_size])
        groups.append(remaining_numbers[group_size:])





    # fiteo el RandomForest con ArbolDecisionID3
    rf = RandomForest(clase_arbol="id3", cantidad_arboles = 10, cantidad_atributos='sqrt', max_prof=10, min_obs_nodo=100)
    rf.fit(x_train, y_train)

    # Predice con el RandomForest
    predicciones = rf.predict(x_test)
    print(f'Accuracy Score: {Metricas.accuracy_score(y_test, predicciones)}')

        