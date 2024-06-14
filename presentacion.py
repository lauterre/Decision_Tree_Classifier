from arbol_clasificador_c45 import ArbolClasificadorC45
from arbol_clasificador_id3 import ArbolClasificadorID3
import pandas as pd
from bosque_clasificador import BosqueClasificador
from herramientas import GridSearch, Herramientas
from metricas import Metricas
from pandas.api.types import CategoricalDtype

def mostrar_id3():
    df = pd.read_csv('datasets/playtennis.csv')
    X = df.drop(columns='Play Tennis')
    y = df['Play Tennis']
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3, random_state=42)
    arbol_id3 = ArbolClasificadorID3()
    arbol_id3.fit(x_train, y_train)
    print(arbol_id3)
    arbol_id3.graficar()
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_id3.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_id3.predict(x_test))}')

def mostrar_c45_titanic():
    titanic = pd.read_csv("./datasets/titanic.csv")
    orden_clases = [1, 2, 3]
    titanic['Pclass'] = titanic['Pclass'].astype(CategoricalDtype(categories=orden_clases, ordered=True))

    X = titanic.drop("Survived", axis = 1)
    y = titanic["Survived"]
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3)

    
    print(titanic.info())
    input('Presione Enter para continuar...')
    arbol_c45 = ArbolClasificadorC45(max_prof = 5)
    arbol_c45.fit(x_train, y_train)
    print(arbol_c45)
    arbol_c45.graficar()
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_c45.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_c45.predict(x_test))}')

def mostrar_c45_tennis():
    tennis = pd.read_csv('datasets/playtennis.csv')
    temperature_order = ['Cool', 'Mild', 'Hot']
    humidity_order = ['Normal', 'High']
    wind_order = ['Weak', 'Strong']
    tennis['Temperature'] = tennis['Temperature'].astype(CategoricalDtype(categories=temperature_order, ordered=True))
    tennis['Humidity'] = tennis['Humidity'].astype(CategoricalDtype(categories=humidity_order, ordered=True))
    tennis['Wind'] = tennis['Wind'].astype(CategoricalDtype(categories=wind_order, ordered=True))
    X = tennis.drop(columns='Play Tennis')
    y = tennis['Play Tennis']
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3, random_state=42)
    arbol_c45 = ArbolClasificadorC45()
    arbol_c45.fit(x_train, y_train)
    print(arbol_c45)
    arbol_c45.graficar()
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_c45.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_c45.predict(x_test))}')

def mostrar_c45_na():
    patientsna = pd.read_csv("./datasets/cancer_patients_con_NA.csv", index_col=0)
    patientsna = patientsna.drop("Patient Id", axis = 1)
    patientsna.loc[:, patientsna.columns != "Age"] = patientsna.loc[:, patientsna.columns != "Age"].astype(str)
    print(patientsna.head())
    X = patientsna.drop("Level", axis = 1)
    y = patientsna["Level"]
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3, random_state=42)
    input('Presione Enter para continuar...')
    arbol_c45 = ArbolClasificadorC45()
    arbol_c45.fit(x_train, y_train)
    print(arbol_c45)
    arbol_c45.graficar()
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_c45.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_c45.predict(x_test), promedio= 'ponderado')}')

def mostrar_bosque():
    patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str)
    print(patients.info())
    input('Presione Enter para continuar...')
    X = patients.drop('Level', axis=1)
    y = patients['Level']

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.15, random_state=42)
    rf = BosqueClasificador(clase_arbol="id3", cantidad_arboles = 10, cantidad_atributos='sqrt', max_prof=2, min_obs_nodo=10, verbose=True)
    print(f'Score promedio: {Herramientas.cross_validation(x_train, y_train, rf, 5,verbose=True)}')

def mostrar_grid_search():
    patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    patients = patients.drop("Patient Id", axis = 1)
    patients.loc[:, patients.columns != "Age"] = patients.loc[:, patients.columns != "Age"].astype(str)

    X = patients.drop("Level", axis=1)
    y = patients["Level"]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    rf = BosqueClasificador()
    grid_search = GridSearch(rf, {'clase_arbol': ['id3', 'c45'],'max_prof': [1,2], 'min_obs_nodo': [10, 50]}, k_fold=3)

    grid_search.fit(x_train, y_train)
    input('Veamos los resultados del Grid Search...')
    print(grid_search.mostrar_resultados())
    mejor_bosque = BosqueClasificador(**grid_search.mejores_params)
    input('Entrenamos el bosque con los mejores parametros...')
    mejor_bosque.fit(x_train, y_train)

    y_pred = mejor_bosque.predict(x_test)
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred)}")
    print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")





def prueba():
    print(f'Bienvenidxs a la presentación del TPI de Algoritmos 2')
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador ID3\nDataset: Playtennis')
    input('Presione Enter para continuar...')
    mostrar_id3()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador C4.5\nDataset: tennis')
    input('Presione Enter para continuar...')
    mostrar_c45_tennis()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador C4.5\nDataset: Titanic')
    mostrar_c45_titanic()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador C4.5\nDataset: Cancer Patients con NA')
    input('Presione Enter para continuar...')
    mostrar_c45_na()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Bosque Clasificar\nDataset: Cancer Patients\nHiperparametros: clase_arbol="id3", cantidad_arboles = 10, cantidad_atributos="sqrt", max_prof=2, min_obs_nodo=10')
    input('Presione Enter para continuar...')
    mostrar_bosque()
    input('Presione Enter para continuar...')
    print(f'Grid Search\nDataset: Cancer Patients\nHiperparametros a testear: clase_arbol=["id3", "c45"], max_prof=[2, 3], min_obs_nodo=[10, 20]')
    mostrar_grid_search()


if __name__ == '__main__':
    mostrar_grid_search()