from src.ArbolDecision.arbol_clasificador_C45 import ArbolClasificadorC45
from src.ArbolDecision.arbol_clasificador_ID3 import ArbolClasificadorID3
import pandas as pd
from src.BosqueClasificador.bosque_clasificador import BosqueClasificador 
from src.tools.herramientas import GridSearch, Herramientas
from src.tools.metricas import Metricas
from pandas.api.types import CategoricalDtype

tennis = pd.read_csv('datasets/PlayTennis.csv')
titanic = pd.read_csv("./datasets/titanic.csv")

patientsna = pd.read_csv("./datasets/cancer_patients_con_NA.csv", index_col=0)


patients = pd.read_csv("./datasets/cancer_patients.csv", index_col=0)
    

def mostrar_id3():
    input('Dataframe PlayTennis:\n')
    print(tennis.head(10))
    X = tennis.drop(columns='Play Tennis')
    y = tennis['Play Tennis']
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3, random_state=42)
    arbol_id3 = ArbolClasificadorID3()
    arbol_id3.fit(x_train, y_train)
    input("Impresión en consola:\n")
    print(arbol_id3)
    input("Graficamos el arbol...\n")
    arbol_id3.graficar_feo()
    input("Métricas del modelo: \n")
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_id3.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_id3.predict(x_test))}')

def mostrar_c45_titanic():
    input('Dataframe Titanic:\n')
    print(titanic.head(10))
    
    X = titanic.drop("Survived", axis = 1)
    y = titanic["Survived"]
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3)
    
    input('Fiteamos el arbol con max_prof = 5 :\n')
    arbol_c45 = ArbolClasificadorC45(max_prof = 5)
    arbol_c45.fit(x_train, y_train)

    input("Impresión en consola:\n")
    print(arbol_c45)
    input("Graficamos el arbol...\n")
    arbol_c45.graficar_feo()
    input("Métricas del modelo: \n")
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_c45.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_c45.predict(x_test))}')

def mostrar_c45_tennis():
    temperature_order = ['Cool', 'Mild', 'Hot']
    humidity_order = ['Normal', 'High']
    wind_order = ['Weak', 'Strong']
    tennis['Temperature'] = tennis['Temperature'].astype(CategoricalDtype(categories=temperature_order, ordered=True))
    tennis['Humidity'] = tennis['Humidity'].astype(CategoricalDtype(categories=humidity_order, ordered=True))
    tennis['Wind'] = tennis['Wind'].astype(CategoricalDtype(categories=wind_order, ordered=True))
    input("Dataframe Tennis ordinal:\n")
    print(tennis.head(10))
    print(tennis.info())

    X = tennis.drop(columns='Play Tennis')
    y = tennis['Play Tennis']
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3, random_state=42)
    arbol_c45 = ArbolClasificadorC45()
    arbol_c45.fit(x_train, y_train)
    input("Impresión en consola:\n")
    print(arbol_c45)
    input("Graficamos el arbol...\n")
    arbol_c45.graficar_feo()
    input("Métricas del modelo: \n")
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_c45.predict(x_test))}')
    print(f'F1 Score: {Metricas.f1_score(y_test, arbol_c45.predict(x_test))}')

def mostrar_c45_na():
    input('Dataframe Cancer Patients con NA:\n')
    print(patientsna.head(10))
    X = patientsna.drop("Level", axis = 1)
    y = patientsna["Level"]
    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.3, random_state=42)
    arbol_c45 = ArbolClasificadorC45()
    arbol_c45.fit(x_train, y_train)
    input("Impresión en consola:\n")
    print(arbol_c45)
    input("Graficamos el arbol...\n")
    arbol_c45.graficar_feo()
    input("Métricas del modelo: \n")
    print(f'Acuraccy Score: {Metricas.accuracy_score(y_test, arbol_c45.predict(x_test))}')
    print(f'F1 Score ponderado: {Metricas.f1_score(y_test, arbol_c45.predict(x_test), promedio= 'ponderado')}')

def mostrar_bosque_id3():
    input('Dataframe Cancer Patients sin NA:\n')
    print(patients.head(10))
    X = patients.drop('Level', axis=1)
    y = patients['Level']

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.15, random_state=42)
    rf = BosqueClasificador(clase_arbol="id3", cantidad_arboles = 10, cantidad_atributos='all', max_prof=2, min_obs_nodo=10, verbose=True)
    print("Hiperparametros del Bosque: \n clase_arbol='id3', cantidad_arboles = 10, cantidad_atributos='all', max_prof=2, min_obs_nodo=10")
    input('Probamos la validación cruzada con 5 folds: \n')
    print(f'CV Score: {Herramientas.cross_validation(x_train, y_train, rf, 5,verbose=True)}')
    print(f"Metrícas del modelo:")
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(f'Accuracy Score: {Metricas.accuracy_score(y_test, y_pred)}')
    print(f'F1 Score ponderado: {Metricas.f1_score(y_test, y_pred, promedio="ponderado")}')

def mostrar_grid_search():
    input("Volvemos a usar cancer patients sin NA")
    X = patients.drop("Level", axis=1)
    y = patients["Level"]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    rf = BosqueClasificador()
    print('Probamos el Grid Search con 3 folds: \n')
    input('Hiperparametros a testear: clase_arbol=["id3", "c45"], max_prof=[2, 3], min_obs_nodo=[10, 50]\n')
    grid_search = GridSearch(rf, {'clase_arbol': ['id3', 'c45'],'max_prof': [2,3], 'min_obs_nodo': [10, 50]}, k_fold=3)

    grid_search.fit(x_train, y_train)
    input('Veamos los resultados del Grid Search...')
    print(grid_search.mostrar_resultados())
    mejor_bosque = BosqueClasificador(**grid_search.mejores_params)
    input('Entrenamos el bosque con los mejores parametros...')
    mejor_bosque.fit(x_train, y_train)

    input('Lo probamos en el set de prueba...\n')
    y_pred = mejor_bosque.predict(x_test)
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred)}")
    print(f"f1-score ponderado en set de prueba: {Metricas.f1_score(y_test, y_pred, promedio='ponderado')}\n")

def mostrar_grid_search_numerico():
    input("Volvemos a usar titanic")
    X = titanic.drop("Survived", axis=1)
    y = titanic["Survived"]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    rf = BosqueClasificador()
    print('Probamos el Grid Search con 3 folds: \n')
    input('Hiperparametros a testear: clase_arbol=["id3", "c45"], max_prof=[2, 3], min_obs_nodo=[10, 50]\n')
    grid_search = GridSearch(rf, {'clase_arbol': ['id3', 'c45'],'max_prof': [2,3], 'min_obs_nodo': [10, 50]}, k_fold=3)

    grid_search.fit(x_train, y_train)
    input('Veamos los resultados del Grid Search...')
    print(grid_search.mostrar_resultados())
    mejor_bosque = BosqueClasificador(**grid_search.mejores_params)
    input('Entrenamos el bosque con los mejores parametros...')
    mejor_bosque.fit(x_train, y_train)

    input('Lo probamos en el set de prueba...\n')
    y_pred = mejor_bosque.predict(x_test)
    print(f"accuracy en set de prueba: {Metricas.accuracy_score(y_test, y_pred)}")
    print(f"f1-score en set de prueba: {Metricas.f1_score(y_test, y_pred)}\n")

def mostrar_cross_validation():
    X = titanic.drop("Survived", axis=1)
    y = titanic["Survived"]

    x_train, x_test, y_train, y_test = Herramientas.dividir_set(X, y, test_size=0.20, random_state=42)
    arbol = ArbolClasificadorC45(max_prof = 3)
    print(f'CV Score: {Herramientas.cross_validation(x_train, y_train, arbol, 5,verbose=True)}')


def prueba():
    print(f'Bienvenidxs a la presentación del TPI de Algoritmos 2')
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador ID3')
    mostrar_id3()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador C4.5')
    input('Presione Enter para continuar...')
    mostrar_c45_tennis()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador C4.5 con Dataset numerico')
    mostrar_c45_titanic()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Arbol clasificador C4.5 con NA')
    input('Presione Enter para continuar...')
    mostrar_c45_na()
    input('Presione Enter para continuar...')
    print(f'Contruccion del Bosque Clasificar ID3')
    input('Presione Enter para continuar...')
    mostrar_bosque_id3()
    input('Presione Enter para continuar...')
    print(f'Grid Search')
    mostrar_grid_search()
    print(f'Grid Search con dataset numerico')
    mostrar_grid_search_numerico()


if __name__ == '__main__':
    # prueba()
    mostrar_cross_validation()