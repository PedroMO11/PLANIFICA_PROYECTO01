"""Sistema adaptativo de decision ante fraude transaccional.

Proyecto del curso Planificacion y Toma de Decisiones en IA (UTEC).
Implementa el plan de PLAN_IMPLEMENTACION.md: integracion de dos fuentes,
features causales, tres familias de modelos, calibracion Platt por version,
politica economica de tres acciones con cupo, y olvido por ventanas fijas
W30/W60/W90 frente a las referencias S0/E15.

Invariantes que el codigo hace cumplir, no solo documenta:

1. Ninguna transformacion se ajusta fuera del tramo de fit de su propia version.
2. Una etiqueta solo es visible cuando available_at < cutoff, con desigualdad estricta.
3. Las features se emiten ANTES de actualizar el estado historico (causalidad).
4. Los cuatro roles temporales por version (predictor, calibrador, politica,
   validacion de promocion) son disjuntos por construccion.
5. El cupo diario de revision se reserva de forma atomica e idempotente.
"""

__version__ = "1.0.0"

SEED = 42

__all__ = ["__version__", "SEED"]
