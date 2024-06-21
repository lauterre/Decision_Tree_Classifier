class LongitudInvalidaException(Exception):
    """Excepción lanzada cuando las longitudes no coinciden."""
    
    def __init__(self, mensaje):
        super().__init__(mensaje)

class ArbolNoEntrenadoException(Exception):
    """Excepción lanzada cuando se intenta predecir sin haber entrenado un árbol."""
    
    def __init__(self, mensaje="Error: No se ha entrenado un árbol"):
        super().__init__(mensaje)

class ArbolEntrenadoException(Exception):
    """Excepción lanzada cuando se intenta entrenar un arbol que ya fue entrenado."""
    
    def __init__(self, mensaje="Error: El arbol ya fue entrenado"):
        super().__init__(mensaje)

class HiperparametroInvalidoException(Exception):
    """Excepción lanzada cuando se intenta usar un hiperparámetro inválido."""
    
    def __init__(self, hiperparametro ,mensaje="Error: Hiperparámetro inválido: "):
        super().__init__(mensaje + hiperparametro)


class BosqueEntrenadoException(Exception):
    """Excepción lanzada cuando se intenta entrenar un bosque que ya fue entrenado."""
    
    def __init__(self, mensaje="Error: El bosque ya fue entrenado"):
        super().__init__(mensaje)

class BosqueNoEntrenadoException(Exception):
    """Excepción lanzada cuando se intenta predecir sin haber entrenado un bosque."""
    
    def __init__(self, mensaje="Error: No se ha entrenado un bosque"):
        super().__init__(mensaje)

class PromedioInvalidoException(Exception):
    """Excepción lanzada cuando se intenta usar un promedio inválido."""
    
    def __init__(self, mensaje):
        super().__init__(mensaje)

class GridSearchNoEntrenadaException(Exception):
    """Excepción lanzada cuando se intenta predecir sin haber entrenado un GridSearch."""
    
    def __init__(self, mensaje="Error: No se ha entrenado un GridSearch"):
        super().__init__(mensaje)  