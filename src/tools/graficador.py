import pydot
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
'''
Modulo para graficar arboles de decision.
Para usar este graficador, es necesario instalar Graphviz en la computadora.
https://graphviz.org/download/
Caso contrario, usar GraficadorArbolFeo
'''

class GraficadorArbol():
    def __init__(self, arbol):
        '''
        Inicializa el graficador de arboles.
        Args:
            arbol (Arbol): Arbol de decision a graficar.
        '''
        self.arbol = arbol
        self.dot = pydot.Dot(graph_type='graph')
        self.colores_clases = self._generar_colores_clases()

    def _generar_colores_clases(self) -> dict[str,str]:
        '''Funcion auxiliar para generar colores para las clases del arbol.
        Returns:
            dict: Diccionario con colores asignados a cada clase.
        '''
        colores = ['lightblue', 'lightgreen', 'lightpink', 'lightsalmon', 'lightyellow', 'lightsteelblue']
        return {clase: colores[i] for i, clase in enumerate(self.arbol.target_categorias)}
    
    def graficar(self) -> None:
        '''Grafica el arbol de decision.'''
        self._agregar_nodos(self.arbol)
        self._agregar_aristas(self.arbol)
        png_data = self.dot.create_png()
        self._mostrar_png(png_data)
    
    def _crear_caja(self, arbol):
        '''Crea el contenido que se muestra en cada nodo del arbol.
        '''
        retorno = []
        if not arbol.es_raiz():
            retorno.append(f"{arbol.atributo_split_anterior}{arbol.signo_split_anterior}{round(arbol.valor_split_anterior, 2) if isinstance(arbol.valor_split_anterior, float) else arbol.valor_split_anterior}")
        retorno.append(f"Muestras: {arbol._total_samples()}")
        retorno.append(f"Conteo: {arbol._values()}")
        retorno.append(f"{arbol.impureza}: {round(arbol.impureza.calcular(arbol.target), 3)}")
        retorno.append(f"Clase: {arbol.clase}")
        return "\n".join(retorno)

    def _agregar_nodos(self, arbol) -> None:
        '''Agrega los nodos al grafo del arbol.
        '''
        nodo_id = str(id(arbol))
        atributos_nodo = {
            "label": self._crear_caja(arbol),
            "shape": "box",
            "style": "rounded, filled",
            "fontname": "Helvetica",
            "fontsize": "15",
        }
        if arbol.es_hoja():
            atributos_nodo["fillcolor"] = self.colores_clases.get(arbol.clase, "white")
        else:
            atributos_nodo["fillcolor"] = "white"
        self.dot.add_node(pydot.Node(nodo_id, **atributos_nodo))
        for subarbol in arbol.subs:
            self._agregar_nodos(subarbol)

    def _agregar_aristas(self, arbol, padre_id: str = "") -> None:
        '''Agrega las aristas al grafo del arbol.'''
        nodo_id = str(id(arbol))
        if padre_id:
            self.dot.add_edge(pydot.Edge(padre_id, nodo_id))
        for subarbol in arbol.subs:
            self._agregar_aristas(subarbol, nodo_id)

    def _mostrar_png(self, png_data: bytes) -> None:
        '''Muestra la imagen del arbol.
        Args:
            png_data (bytes): Imagen del arbol en formato png.
        '''
        image = Image.open(BytesIO(png_data))
        plt.imshow(image)
        plt.axis('off')
        plt.show()



class GraficadorArbolFeo:
    def __init__(self, arbol , ax=None, fontsize=None):
        '''Constructor del graficador de arboles.
        Args:
            arbol (Arbol): Arbol de decision a graficar.
            ax (matplotlib.axes._subplots.AxesSubplot): Eje donde se graficara el arbol.
            fontsize (int): Tamaño de la fuente en el grafico.
        '''
        self.arbol = arbol
        self.ax = ax
        self.fontsize = fontsize

    def plot(self):
        '''Grafica el arbol de decision.'''
        if self.ax is None:
            _, self.ax = plt.subplots(figsize=(self._get_fig_size()))
        self.ax.clear()
        self.ax.set_axis_off()
        self._plot_tree(self.arbol, xy=(self.arbol.ancho_nivel(), self.arbol.altura()), level_width=self.arbol.ancho_nivel())
        plt.show()
        
    def _plot_tree(self, arbol, xy, level_width, level_height=1, depth=0):
        '''Funcion interna para graficar el arbol.
        Args:
            arbol (Arbol): Arbol de decision a graficar.
            xy (tuple): Coordenadas del nodo actual.
            level_width (int): Ancho del nivel actual.
            level_height (int): Altura del nivel actual.
            depth (int): Profundidad del nodo actual.
        '''
        
        bbox_args = dict(boxstyle="round", fc=self.get_color(arbol, arbol.clase) if arbol.es_hoja() else "white", ec="black")
        caja = self._crear_caja(arbol)
        self._annotate(caja, xy, depth, bbox_args)

        if arbol.subs:
            num_subs = len(arbol.subs)
            width_per_node = level_width / max(num_subs, 1)
            child_positions = []

            for i, subarbol in enumerate(arbol.subs):
                new_x = xy[0] - (level_width / 2) + (i + 0.5) * width_per_node
                new_xy = (new_x, xy[1] - level_height)
                child_positions.append((subarbol, new_xy))
                self.ax.plot([xy[0], new_x], [xy[1], new_xy[1]], color="black")

            for subarbol, new_xy in child_positions:
                self._plot_tree(subarbol, new_xy, width_per_node, level_height, depth + 1)
        
    def _crear_caja(self, arbol):
        '''Crea el contenido que se muestra en cada nodo del arbol.
        Args:
            arbol (Arbol): Arbol de decision.
        Returns:
            str: Contenido del nodo.
        '''
        retorno = []
        if not arbol.es_raiz():
            retorno.append(f"{arbol.atributo_split_anterior}{arbol.signo_split_anterior}{arbol.valor_split_anterior}")
        retorno.append(f"Muestras: {arbol._total_samples()}")
        retorno.append(f"Conteo: {arbol._values()}")
        retorno.append(f"{arbol.criterio_impureza}: {round(arbol._impureza(), 3)}")
        retorno.append(f"Clase: {arbol.clase}")
        return "\n".join(retorno)
        
    def _annotate(self, text, xy, depth, bbox_args):
        '''Agrega el texto al grafico.
        Args:
            text (str): Texto a agregar.
            xy (tuple): Coordenadas del texto.
            depth (int): Profundidad del nodo.
            bbox_args (dict): Argumentos para el cuadro de texto.
        '''
        dynamic_fontsize = self._calculate_fontsize()
        kwargs = dict(
            bbox=bbox_args,
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="data",
            fontsize=dynamic_fontsize
        )
        self.ax.annotate(text, xy=xy, **kwargs)
        
    def _calculate_fontsize(self):
        '''Calcula el tamaño de la fuente en base a la cantidad de nodos del arbol.
        Returns:
            int: Tamaño de la fuente.
        ''' 
        num_nodes = len(self.arbol)
        base_size = 10  # base fontsize
        return max(2, base_size - num_nodes // 10)
        
    def _get_fig_size(self):
        '''Calcula el tamaño de la figura en base a la altura y ancho del arbol.
        Returns:
            tuple: Tamaño de la figura.
        '''
        prof = self.arbol.altura()
        ancho_max = self.arbol.ancho()
        ancho = max(1, ancho_max * 2)
        altura = max(1, prof)
        return (ancho, altura)
        
    def get_color(self, arbol, clase):
        '''Devuelve el color asignado a una clase.
        
        Args:
            arbol (Arbol): Arbol de decision.
            clase (str): Clase a colorear.

        Returns:
            str: Color asignado a la clase.
        '''
        clases = arbol.target_categorias
        colores = ["lightgreen", "lightblue", (1, 1, 0.6)]
        colores_clases = {c: colores[i] for i, c in enumerate(clases)}
        return colores_clases.get(clase, "white")

        