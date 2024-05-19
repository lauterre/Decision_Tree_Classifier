from _superclases import Arbol
import matplotlib.pyplot as plt


class TreePlot:
    def __init__(self, arbol: Arbol, ax=None, fontsize=None, filled=False):
        self.arbol = arbol
        self.ax = ax or plt.gca()
        self.fontsize = fontsize
        self.filled = filled
        self.bbox_args = dict(boxstyle="round", fc="white", ec="black")
        self.line_args = dict(color="black")

    def plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        profundidad_total = self._calcular_profundidad(self.arbol)
        altura = 1.0 / (profundidad_total + 1)
        self._plot_tree(self.arbol, xy_padre=(0.5, 1.0), ancho=1.0, altura=altura, profundidad=0)
        plt.show()

    def _plot_tree(self, arbol: Arbol, xy_padre, ancho, altura, profundidad):
        # Calcular la posición del nodo actual
        xy_actual = (xy_padre[0], xy_padre[1] - altura)
        
        # Anotar el nodo actual
        caja = self._crear_caja(arbol, profundidad == 0)
        self._annotate(caja, xy_actual, profundidad)
        
        if arbol.subs:
            num_subarboles = len(arbol.subs)
            ancho_nodo = ancho / max(num_subarboles, 1)
            for i, subarbol in enumerate(arbol.subs):
                new_x = xy_padre[0] - (ancho / 2) + (i + 0.5) * ancho_nodo
                new_xy = (new_x, xy_actual[1])
                # Dibujar una línea desde el nodo padre al nodo hijo
                self.ax.plot([xy_actual[0], new_x], [xy_actual[1], new_xy[1] - altura], **self.line_args)
                self._plot_tree(subarbol, new_xy, ancho_nodo, altura, profundidad + 1)

    def _crear_caja(self, arbol: Arbol, es_raiz: bool) -> str:
        retorno = []
        if arbol.atributo:
            retorno.append(f"Atributo: {arbol.atributo}")
        if arbol.categoria:
            retorno.append(f"Categoría: {arbol.categoria}")
        retorno.append(f"Entropía: {arbol.entropia():.2f}")
        retorno.append(f"Muestras: {arbol._total_samples()}")
        retorno.append(f"Valores: {arbol._values()}")
        if not arbol.subs and arbol.clase:
            retorno.append(f"Clase: {arbol.clase}")
        return "\n".join(retorno)

    def _annotate(self, text, xy, profundidad):
        kwargs = dict(
            bbox=self.bbox_args,
            ha="center",
            va="center",
            zorder=100 - 10 * profundidad,
            xycoords="axes fraction"
        )
        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        self.ax.annotate(text, xy=xy, **kwargs)

    def _calcular_profundidad(self, arbol: Arbol) -> int:
        if not arbol.subs:
            return 0
        return 1 + max(self._calcular_profundidad(sub) for sub in arbol.subs)