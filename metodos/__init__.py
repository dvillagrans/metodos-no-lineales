"""
Módulo de métodos de optimización
Contiene implementaciones de diferentes algoritmos de optimización no lineal
"""

from .gradient_descent import GradientDescentOptimizer
from .newton_method import NewtonOptimizer
from .quasi_newton import QuasiNewtonOptimizer
from .line_search import LineSearchOptimizer
from .conjugate_gradient import ConjugateGradientOptimizer
from .constrained_optimization import ConstrainedOptimizer
from .penalty_barrier import PenaltyBarrierOptimizer
from .lagrange_multipliers import LagrangeMultiplierOptimizer

__all__ = [
    'GradientDescentOptimizer',
    'NewtonOptimizer', 
    'QuasiNewtonOptimizer',
    'LineSearchOptimizer',
    'ConjugateGradientOptimizer',
    'ConstrainedOptimizer',
    'PenaltyBarrierOptimizer',
    'LagrangeMultiplierOptimizer'
]
