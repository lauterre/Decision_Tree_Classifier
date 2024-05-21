from _superclases import Arbol
import matplotlib.pyplot as plt
from collections import defaultdict  

class TreePlot:
    def __init__(self, arbol, ax=None, fontsize=0.1):
        self.arbol = arbol
        self.ax = ax
        self.fontsize = fontsize

    def plot(self):
        if self.ax is None:
            _, self.ax = plt.subplots(figsize=(self._ancho_max(self.arbol), self.arbol.altura())) #self.get_fig_size()
        self.ax.clear()
        self.ax.set_axis_off()
        xlim = (-1, 1) # deberia depender del ancho max
        ylim = (-1, 1) # deberia depender de la altura
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self._plot_tree(self.arbol, xy=(0, 1), level_width=len(xlim), level_height = len(ylim) *3/10, depth=0) # level_height = 0.6, width = 2
        plt.show()

    def _plot_tree(self, arbol, xy, level_width, level_height, depth):

        bbox_args = dict(boxstyle="round", fc=self.get_color(arbol, arbol.clase) if arbol.es_hoja() else "white", ec="black")
        caja = self._crear_caja(arbol, arbol.es_raiz())
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

    def _crear_caja(self, arbol: Arbol, es_raiz: bool) -> str:
        retorno = []
        if arbol.categoria:
            retorno.append(f"Categoría: {arbol.categoria}")
        retorno.append(f"Muestras: {arbol._total_samples()}")
        retorno.append(f"Valores: {arbol._values()}")
        retorno.append(f"Entropía: {arbol.entropia():.2f}")
        if arbol.atributo:
            retorno.append(f"Atributo: {arbol.atributo}")
        if not arbol.subs and arbol.clase:
            retorno.append(f"Clase: {arbol.clase}")
        return "\n".join(retorno)

    def _annotate(self, text, xy, depth, bbox_args):
        kwargs = dict(
            bbox=bbox_args,
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="data",
            fontsize = self.fontsize
        )
        self.ax.annotate(text, xy=xy, **kwargs)

    # def _get_fig_size(self):
    #     prof = self.arbol.altura()
    #     ancho_max = self._ancho_max(self.arbol)
    #     ancho = max(10, ancho_max * 2)
    #     altura = max(10, prof * 2)
    #     return (ancho, altura)

    def _ancho_max(self, arbol, nivel=0, nodos=None): # en Arbol
        if nodos is None:
            nodos = defaultdict(int)
        nodos[nivel] += 1
        for subarbol in arbol.subs:
            self._ancho_max(subarbol, nivel + 1, nodos)
        return max(nodos.values())
    
    # Podria servir
    def _ancho_nivel(self, arbol, nivel = 0, nodos= None): # deberia estar en Arbol
        if nodos is None:
            nodos = defaultdict(int)
        nodos[nivel] += 1
        for subarbol in arbol.subs:
            self._ancho_nivel(subarbol, nivel + 1, nodos)
        return nodos[nivel]
    
    def get_color(self, arbol, clase):
        clases = arbol.target_categorias
        colores = ["lightgreen", "lightblue", (1, 1, 0.6)] #agregar
        colores_clases = {}
        for i, c in enumerate(clases):
            colores_clases[c] = colores[i]
        return colores_clases[clase]
