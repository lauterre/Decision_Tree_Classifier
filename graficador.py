import pydot
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

class GraficadorArbol():
    def __init__(self, arbol):
        self.arbol = arbol
        self.dot = pydot.Dot(graph_type='graph')
        self.colores_clases = self._generar_colores_clases()

    def _generar_colores_clases(self):
        colores = ['lightblue', 'lightgreen', 'lightpink', 'lightsalmon', 'lightyellow', 'lightsteelblue']
        return {clase: colores[i] for i, clase in enumerate(self.arbol.target_categorias)}
    
    def graficar(self) -> None:
        self._agregar_nodos(self.arbol)
        self._agregar_aristas(self.arbol)
        png_data = self.dot.create_png()
        self._mostrar_png(png_data)
    
    def _crear_caja(self, arbol):
        retorno = []
        if not arbol.es_raiz():
            retorno.append(f"{arbol.atributo_split_anterior}{arbol.signo_split_anterior}{arbol.valor_split_anterior}")
        retorno.append(f"Muestras: {arbol._total_samples()}")
        retorno.append(f"Conteo: {arbol._values()}")
        retorno.append(f"{arbol.impureza}: {round(arbol.impureza.calcular(arbol.target), 3)}")
        retorno.append(f"Clase: {arbol.clase}")
        return "\n".join(retorno)

    def _agregar_nodos(self, arbol) -> None:
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
        nodo_id = str(id(arbol))
        if padre_id:
            self.dot.add_edge(pydot.Edge(padre_id, nodo_id))
        for subarbol in arbol.subs:
            self._agregar_aristas(subarbol, nodo_id)

    def _mostrar_png(self, png_data: bytes) -> None:
        image = Image.open(BytesIO(png_data))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        