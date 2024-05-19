from abc import ABC, abstractmethod
from typing import Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Clasificador(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

class ClasificadorArbol(Clasificador, ABC):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1):
        self.max_prof = max_prof
        self.min_obs_nodo = min_obs_nodo
    
class Arbol(ABC):
    def __init__(self) -> None:
        self.data: pd.DataFrame 
        self.target: pd.Series
        self.atributo: Optional[str] = None
        self.valor: Optional[str]= None
        self.target_categorias: Optional[list[str]]= None
        self.clase: Optional[str] = None
        self.subarboles: list[Arbol]= []
    
    def es_raiz(self):
        return self.valor is None
    
    def es_hoja(self):
        return self.subarboles == []
    
    def __len__(self) -> int:
        if self.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.subarboles])
        
    def _values(self):
        recuento_values = self.target.value_counts()
        values = []
        for valor in self.target_categorias:
            value = recuento_values.get(valor, 0)
            values.append(value)
        return values
    
    def altura(self) -> int:
        altura_actual = 0
        for subarbol in self.subarboles:
            altura_actual = max(altura_actual, subarbol.altura())
        return altura_actual + 1

    def _total_samples(self):
        return len(self.data)
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def _mejor_split(self):
        raise NotImplementedError
    
    @abstractmethod
    def _split(self, atributo: str, valor: Any) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def entropia(self):
        raise NotImplementedError
    
    @abstractmethod
    def _information_gain(self, atributo: str) -> float:
        raise NotImplementedError 
    @abstractmethod
    def imprimir(self) -> None:
        raise NotImplementedError   


class ArbolID3(Arbol, ClasificadorArbol):
    def __init__(self, max_prof: int = -1, min_obs_nodo: int = -1) -> None:
        super().__init__()
        ClasificadorArbol.__init__(self, max_prof, min_obs_nodo)
        
    def _traer_hiperparametros(self, arbol_previo):
        self.max_prof = arbol_previo.max_prof
        self.min_obs_nodo = arbol_previo.min_obs_nodo
    
    def _mejor_split(self) -> str: 
        mejor_ig = -1
        mejor_atributo = None
        atributos = self.data.columns

        for atributo in atributos:
            ig = self._information_gain(atributo)
            if ig > mejor_ig:
                mejor_ig = ig
                mejor_atributo = atributo
        
        return mejor_atributo
    
    def copy(self):
        nuevo = ArbolID3(self.max_prof, self.min_obs_nodo)
        nuevo.data = self.data.copy()
        nuevo.target = self.target.copy()
        nuevo.atributo = self.atributo
        nuevo.valor = self.valor
        nuevo.target_categorias = self.target_categorias.copy() 
        nuevo.clase = self.clase
        nuevo.subarboles = [sub.copy() for sub in self.subarboles]
        return nuevo

    def _split(self, atributo: str) -> None:
        self.atributo = atributo # guardo el atributo por el cual spliteo
        for categoria in self.data[atributo].unique():
            nueva_data = self.data[self.data[atributo] == categoria]
            nueva_data = nueva_data.drop(atributo, axis = 1) # la data del nuevo nodo sin el atributo por el cual ya se filtró
            nuevo_target = self.target[self.data[atributo] == categoria]
            nuevo_arbol = ArbolID3()
            nuevo_arbol.data = nueva_data
            nuevo_arbol.target = nuevo_target
            nuevo_arbol.valor = categoria
            nuevo_arbol.clase = nuevo_target.value_counts().idxmax()
            nuevo_arbol._traer_hiperparametros(self) # hice un metodo porque van a ser muchos de hiperparametros
            self.subarboles.append(nuevo_arbol)
    
    def entropia(self) -> float:
        entropia = 0
        proporciones = self.target.value_counts(normalize= True)
        target_categorias = self.target.unique()
        for c in target_categorias:
            proporcion = proporciones.get(c, 0)
            entropia += proporcion * np.log2(proporcion)
        return -entropia if entropia != 0 else 0
    
    def _information_gain(self, atributo: str) -> float:
        entropia_actual = self.entropia()
        len_actual = len(self.data)

        nuevo = self.copy()
        nuevo._split(atributo)

        entropias_subarboles = 0 

        for subarbol in nuevo.subarboles:
            entropia = subarbol.entropia()
            len_subarbol = len(subarbol.data)
            entropias_subarboles += ((len_subarbol/len_actual)*entropia)

        information_gain = entropia_actual - entropias_subarboles
        return information_gain       


    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        Condicion de split
              - Unico valor para target (nodo puro)
              - No hay mas atributos
              - max_profundidaself.data = X
        '''
        self.target = y
        self.data = X
        self.clase = self.target.value_counts().idxmax()
        
        def _interna(arbol: ArbolID3, prof_acum: int = 0):
            arbol.target_categorias = y.unique()
            
            if prof_acum == 0:
                prof_acum = 1
            
            if not ( len(arbol.target.unique()) == 1 or len(arbol.data.columns) == 0 
                    or (arbol.max_prof != -1 and arbol.max_prof <= prof_acum) 
                    or (arbol.min_obs_nodo != -1 and arbol.min_obs_nodo > arbol._total_samples() ) ):
                
                mejor_atributo = arbol._mejor_split()
                arbol._split(mejor_atributo)
                for sub_arbol in arbol.subarboles:
                    _interna(sub_arbol, prof_acum+1)

        _interna(self)
    
    def predict(self, X:pd.DataFrame) -> list[str]:
        predicciones = []

        def _recorrer(arbol, fila: pd.Series) -> None:
            if arbol.es_hoja():
                predicciones.append(arbol.clase)
            else:
                direccion = fila[arbol.atributo]
                for subarbol in arbol.subarboles:
                    if direccion == subarbol.categoria: #subarbol.valor
                        _recorrer(subarbol, fila)
        
        for _, fila in X.iterrows():
            _recorrer(self, fila)
        
        return predicciones
    
    def imprimir(self, prefijo: str = '  ', es_ultimo: bool = True) -> None:
        simbolo_rama = '└─── ' if es_ultimo else '├─── '
        split = "Split: " + str(self.atributo)
        rta = "Valor: " + str(self.valor)
        entropia = f"Entropia: {round(self.entropia(), 2)}"
        samples = f"Samples: {str (self._total_samples())}"
        values = f"Values: {str(self._values())}"
        clase = 'Clase: ' + str(self.clase)
        if self.es_raiz():
            print(entropia)
            print(samples)
            print(values)
            print(clase)
            print(split)

            for i, sub_arbol in enumerate(self.subarboles):
                ultimo: bool = i == len(self.subarboles) - 1
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
            for i, sub_arbol in enumerate(self.subarboles):
                ultimo: bool = i == len(self.subarboles) - 1
                sub_arbol.imprimir(prefijo, ultimo)
        else:
            prefijo_hoja = prefijo + " "*len(simbolo_rama) if es_ultimo else prefijo + "│" + " "*(len(simbolo_rama) -1)
            print(prefijo + "│")
            print(prefijo + simbolo_rama + rta)
            print(prefijo_hoja + entropia)
            print(prefijo_hoja + samples)
            print(prefijo_hoja + values)
            print(prefijo_hoja + clase)

    def plot_tree(self, ax=None, fontsize=None):
        TreePlot(self, ax, fontsize).plot()

    def __str__(self):
        split = f"Split: {str(self.atributo)}\n" if self.atributo else ""
        valor = f"Valor: {str(self.valor)}\n" if self.valor else ""
        entropia = f"Entropia: {round(self.entropia(), 2)}\n"
        samples = f"Samples: {str (self._total_samples())}\n"
        values = f"Values: {str(self._values())}\n"
        clase = f"Clase: {str(self.clase)}"
        return f"{split}{valor}{entropia}{samples}{values}{clase}"
    
    def altura(self) -> int:
        if self.es_hoja():
            return 1
        return 1 + max(subarbol.altura() for subarbol in self.subarboles)

class TreePlot:
    def __init__(self, tree: ArbolID3, ax=None, fontsize=10):
        self.tree = tree
        self.ax = ax
        self.fontsize = fontsize

    def plot(self):
        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=self._get_fig_size())
        self.ax.clear()
        self.ax.set_axis_off()
        self._plot_tree(self.tree, parent_xy=(0.5, 1.0), level_width=1.0, level_height=0.2, depth=0)
        plt.show()

    def _plot_tree(self, tree, parent_xy, level_width, level_height, depth):
        # Data para la caja
        bbox_args = dict(boxstyle="round", fc="lightgreen" if tree.es_hoja() else "white", ec="black")

        # Calculate the position of the current node
        xy = (parent_xy[0], parent_xy[1] - level_height)

        # Anotar el nodo
        self._annotate(str(tree), xy, depth, bbox_args) # str(tree) es el texto de la caja

        if tree.subarboles:
            num_subarboles = len(tree.subarboles)
            width_per_node = level_width / max(num_subarboles, 1)
            for i, subarbol in enumerate(tree.subarboles):
                new_x = parent_xy[0] - (level_width / 2) + (i + 0.5) * width_per_node
                new_xy = (new_x, xy[1] - level_height)
                # Dibuja la linea (en cualquier parte)
                self.ax.plot([parent_xy[0], new_x], [parent_xy[1], new_xy[1]], color="black")
                self._plot_tree(subarbol, new_xy, width_per_node, level_height, depth + 1)

    def _annotate(self, text, xy, depth, bbox_args):
        kwargs = dict(
            bbox=bbox_args,
            ha="center", # posicion del texto en la caja
            va="bottom", # posicion de la caja, hay que ver como hacer que la linea este en esta posicion
            zorder=100 - 10 * depth,
            xycoords="axes fraction"
        )
        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        self.ax.annotate(text, xy=xy, **kwargs)

    def _get_fig_size(self):
        # Calcular la altura y el ancho del arbol
        depth = self.tree.altura()
        max_width = self._max_width(self.tree)
        # Ajusta la figura en base a esas mediciones
        width = max(10, max_width * 2)  # Minimum width of 10, scales with the tree width
        height = max(10, depth * 2)  # Minimum height of 10, scales with the tree depth
        return (width, height)

    def _max_width(self, tree, level=0, level_counts=None):
        if level_counts is None:
            level_counts = {}
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
        for subarbol in tree.subarboles:
            self._max_width(subarbol, level + 1, level_counts)
        return max(level_counts.values())

# Ejemplo de uso
if __name__ == "__main__":
    tennis = pd.read_csv("PlayTennis.csv")
    patients = pd.read_csv("cancer_patients.csv", index_col=0)

    patients = patients.drop("Patient Id", axis = 1)
    bins = [0, 15, 20, 30, 40, 50, 60, 70, float('inf')]
    labels = ['0-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
    patients['Age'] = pd.cut(patients['Age'], bins=bins, labels=labels, right=False)
    X = patients.drop("Level", axis = 1)
    y = patients["Level"]

    arbol = ArbolID3()
    arbol2 = ArbolID3()

    X2 = tennis.drop("Play Tennis", axis=1)
    y2 = tennis["Play Tennis"]
    
    arbol.fit(X, y)
    arbol2.fit(X2, y2)
    
    arbol.plot_tree()
    arbol2.plot_tree()