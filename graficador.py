import matplotlib.pyplot as plt

class GraficadorArbol:
    def __init__(self, arbol, ax=None, fontsize=None):
        self.arbol = arbol
        self.ax = ax
        self.fontsize = fontsize

    def plot(self):
        if self.ax is None:
            _, self.ax = plt.subplots(figsize=(self._get_fig_size()))
        self.ax.clear()
        self.ax.set_axis_off()
        self._plot_tree(self.arbol, xy=(self.arbol.ancho_nivel(), self.arbol.altura()), level_width=self.arbol.ancho_nivel())
        plt.show()
        
    def _plot_tree(self, arbol, xy, level_width, level_height=1, depth=0):
        
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
        retorno = []
        if not arbol.es_raiz():
            retorno.append(f"{arbol.atributo_split_anterior}{arbol.signo_split_anterior}{arbol.valor_split_anterior}")
        retorno.append(f"Muestras: {arbol._total_samples()}")
        retorno.append(f"Conteo: {arbol._values()}")
        retorno.append(f"{arbol.criterio_impureza}: {arbol._impureza()}")
        retorno.append(f"Clase: {arbol.clase}")
        return "\n".join(retorno)
        
    def _annotate(self, text, xy, depth, bbox_args):
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
        num_nodes = len(self.arbol)
        base_size = 10  # base fontsize
        return max(2, base_size - num_nodes // 10)
        
    def _get_fig_size(self):
        prof = self.arbol.altura()
        ancho_max = self.arbol.ancho()
        ancho = max(1, ancho_max * 2)
        altura = max(1, prof)
        return (ancho, altura)
        
    def get_color(self, arbol, clase):
        clases = arbol.target_categorias
        colores = ["lightgreen", "lightblue", (1, 1, 0.6)]
        colores_clases = {c: colores[i] for i, c in enumerate(clases)}
        return colores_clases.get(clase, "white")
        