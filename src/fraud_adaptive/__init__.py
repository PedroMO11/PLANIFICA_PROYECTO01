"""Sistema adaptativo de decision ante fraude transaccional.

Proyecto del curso Planificacion y Toma de Decisiones en IA (UTEC): integracion de
dos fuentes, features causales, tres familias de modelos, calibracion Platt por
version, politica economica de tres acciones con cupo y adaptacion por
reentrenamiento periodico y ventanas deslizantes.

Invariantes que verifican las pruebas:

1. Ninguna transformacion se ajusta fuera del tramo de fit de su version.
2. Una etiqueta es visible solo cuando available_at < job_time.
3. Las features se emiten antes de actualizar el estado historico.
4. Los roles temporales de cada version (predictor, calibrador y validacion de
   promocion) son disjuntos.
5. El cupo diario de revision se reserva de forma atomica e idempotente.
"""

__version__ = "1.0.0"

SEED = 42

__all__ = ["__version__", "SEED"]
