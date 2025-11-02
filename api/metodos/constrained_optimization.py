"""
Métodos de Optimización con Restricciones
Implementa métodos primales (gradiente restringido)
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class ConstrainedOptimizer:
    """Implementa métodos de optimización con restricciones de igualdad y desigualdad"""
    
    def __init__(self):
        self.available_methods = [
            'projected_gradient',
            'reduced_gradient',
            'constrained_steepest_descent',
            'feasible_direction_method'
        ]
    
    def projected_gradient(self, func: Callable, grad: Callable, x0: np.ndarray,
                          constraints: Optional[Dict] = None,
                          learning_rate: float = 0.01,
                          max_iter: int = 1000, 
                          tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de gradiente proyectado para restricciones de igualdad y desigualdad
        
        Args:
            func: Función objetivo
            grad: Gradiente de la función objetivo
            x0: Punto inicial
            constraints: Diccionario con información de restricciones
            learning_rate: Tasa de aprendizaje
            max_iter: Máximo número de iteraciones
            tol: Tolerancia para convergencia
        """
        if x0 is None:
            raise ValueError("El punto inicial x0 no puede ser None")
            
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        violations = []
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Aplicar proyección según el tipo de restricción
            projected_gradient = gradient
            
            if constraints:
                if constraints['type'] == 'eq':
                    # Para restricciones de igualdad: proyectar el gradiente
                    jac = constraints['jac'](x)
                    jac = jac.reshape(1, -1) if jac.ndim == 1 else jac
                    
                    # Proyector ortogonal al espacio nulo de A
                    try:
                        AAT_inv = np.linalg.pinv(jac @ jac.T)
                        P = np.eye(len(x)) - jac.T @ AAT_inv @ jac
                        projected_gradient = P @ gradient
                    except np.linalg.LinAlgError:
                        projected_gradient = gradient
                        
                elif constraints['type'] == 'ineq':
                    # Para desigualdades: usar método de proyección simple
                    projected_gradient = gradient
            
            # Actualizar posición
            x_new = x - learning_rate * projected_gradient
            
            # Proyectar sobre restricciones si es necesario
            if constraints:
                if constraints['type'] == 'eq':
                    # Proyectar usando Newton para h(x) = 0
                    x_new = self._project_onto_equality_constraint(x_new, constraints)
                elif constraints['type'] == 'ineq':
                    # Proyectar sobre región factible
                    x_new = self._project_onto_feasible_region(x_new, constraints)
            
            x = x_new
            path.append(x.copy())
            errors.append(func(*x))
            
            # Calcular violación de restricciones
            violation = 0
            if constraints:
                constraint_val = constraints['fun'](x)
                if constraints['type'] == 'eq':
                    violation = abs(constraint_val)
                elif constraints['type'] == 'ineq':
                    violation = max(0, -constraint_val)  # violación si constraint_val < 0
            violations.append(violation)
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'violations': violations,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'projected_gradient'
        }
    
    def reduced_gradient(self, func: Callable, grad: Callable, x0: np.ndarray,
                        constraints: Optional[Dict] = None,
                        basic_vars: List[int] = None,
                        learning_rate: float = 0.01,
                        max_iter: int = 1000,
                        tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de gradiente reducido para restricciones de igualdad
        Mejorada: proyecta el punto inicial y cada iterado sobre la restricción de igualdad.
        """
        if x0 is None:
            raise ValueError("El punto inicial x0 no puede ser None")

        x = x0.copy()
        n = len(x)

        # Si no hay restricción de igualdad, usar gradiente proyectado
        if not constraints or constraints['type'] != 'eq':
            return self.projected_gradient(func, grad, x0, constraints, learning_rate, max_iter, tol)

        # Proyectar el punto inicial sobre la restricción de igualdad
        x = self._project_onto_equality_constraint(x, constraints)

        path = [x.copy()]
        errors = [func(*x)]
        violations = []

        for i in range(max_iter):
            gradient = grad(*x)

            # Proyectar el gradiente sobre el subespacio tangente a la restricción
            jac = constraints['jac'](x)
            jac = jac.reshape(1, -1) if jac.ndim == 1 else jac
            try:
                AAT_inv = np.linalg.pinv(jac @ jac.T)
                P = np.eye(n) - jac.T @ AAT_inv @ jac
                direction = -P @ gradient
            except np.linalg.LinAlgError:
                direction = -gradient

            # Si la dirección es muy pequeña, convergió
            if np.linalg.norm(direction) < tol:
                break

            # Paso en la dirección reducida
            x_new = x + learning_rate * direction
            # Proyectar el nuevo punto sobre la restricción de igualdad
            x_new = self._project_onto_equality_constraint(x_new, constraints)

            x = x_new
            path.append(x.copy())
            errors.append(func(*x))

            # Calcular violación de la restricción
            violation = 0
            if constraints:
                constraint_val = constraints['fun'](x)
                if constraints['type'] == 'eq':
                    violation = abs(constraint_val)
                elif constraints['type'] == 'ineq':
                    violation = max(0, -constraint_val)
            violations.append(violation)

        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'violations': violations,
            'iterations': i + 1,
            'converged': np.linalg.norm(direction) < tol,
            'method': 'reduced_gradient'
        }
    
    def constrained_steepest_descent(self, func: Callable, grad: Callable,
                                   constraints_ineq: List[Callable] = None,
                                   constraints_ineq_jac: List[Callable] = None,
                                   x0: np.ndarray = None,
                                   learning_rate: float = 0.01,
                                   max_iter: int = 1000,
                                   tol: float = 1e-6) -> Dict[str, Any]:
        """
        Descenso más pronunciado con restricciones de desigualdad g(x) <= 0
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        active_sets = []
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            # Identificar restricciones activas
            active_constraints = []
            active_gradients = []
            
            if constraints_ineq and constraints_ineq_jac:
                for j, (g, g_jac) in enumerate(zip(constraints_ineq, constraints_ineq_jac)):
                    if g(*x) >= -1e-6:  # Restricción activa
                        active_constraints.append(j)
                        active_gradients.append(g_jac(*x))
            
            active_sets.append(active_constraints.copy())
            
            if len(active_gradients) > 0:
                A = np.array(active_gradients)
                # Proyector sobre espacio factible
                try:
                    AAT_inv = np.linalg.pinv(A @ A.T)
                    P = np.eye(len(x)) - A.T @ AAT_inv @ A
                    feasible_direction = -P @ gradient
                except np.linalg.LinAlgError:
                    feasible_direction = -gradient
            else:
                feasible_direction = -gradient
            
            if np.linalg.norm(feasible_direction) < tol:
                break
            
            # Búsqueda de línea en dirección factible
            alpha = self._line_search_feasible(func, constraints_ineq, x, feasible_direction)
            x = x + alpha * feasible_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'active_sets': active_sets,
            'iterations': i + 1,
            'converged': np.linalg.norm(feasible_direction if 'feasible_direction' in locals() else gradient) < tol,
            'method': 'constrained_steepest_descent'
        }
    
    def feasible_direction_method(self, func: Callable, grad: Callable,
                                 constraints_ineq: List[Callable],
                                 constraints_ineq_jac: List[Callable],
                                 x0: np.ndarray,
                                 max_iter: int = 1000,
                                 tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de direcciones factibles (Zoutendijk)
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            # Resolver subproblema de programación lineal para encontrar dirección factible
            direction = self._solve_direction_subproblem(gradient, constraints_ineq, 
                                                       constraints_ineq_jac, x)
            
            if np.linalg.norm(direction) < tol:
                break
            
            # Búsqueda de línea
            alpha = self._line_search_feasible(func, constraints_ineq, x, direction)
            x = x + alpha * direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(direction if 'direction' in locals() else gradient) < tol,
            'method': 'feasible_direction_method'
        }
    
    def _project_onto_constraints(self, x: np.ndarray, constraints: List[Callable],
                                 jacobians: List[Callable], max_iter: int = 10) -> np.ndarray:
        """Proyecta un punto sobre las restricciones usando método de Newton"""
        x_proj = x.copy()
        
        for _ in range(max_iter):
            h_vals = np.array([h(*x_proj) for h in constraints])
            if np.linalg.norm(h_vals) < 1e-8:
                break
            
            J = np.array([jac(*x_proj) for jac in jacobians])
            try:
                delta = np.linalg.pinv(J) @ h_vals
                x_proj = x_proj - delta
            except np.linalg.LinAlgError:
                break
        
        return x_proj
    
    def _project_onto_equality_constraint(self, x: np.ndarray, constraints: Dict) -> np.ndarray:
        """Proyecta un punto sobre una restricción de igualdad usando Newton"""
        x_proj = x.copy()
        
        for _ in range(10):  # máximo 10 iteraciones
            h_val = constraints['fun'](x_proj)
            if abs(h_val) < 1e-8:
                break
            
            jac = constraints['jac'](x_proj)
            if np.linalg.norm(jac) < 1e-12:
                break
                
            # Un paso de Newton para h(x) = 0
            step = h_val / (jac @ jac)
            x_proj = x_proj - step * jac
        
        return x_proj
    
    def _project_onto_feasible_region(self, x: np.ndarray, constraints: Dict) -> np.ndarray:
        """Proyecta un punto sobre la región factible para restricciones de desigualdad"""
        constraint_val = constraints['fun'](x)
        
        if constraint_val >= 0:
            # Ya está en la región factible
            return x
        
        # Proyección sobre la frontera usando gradiente
        jac = constraints['jac'](x)
        if np.linalg.norm(jac) < 1e-12:
            return x
            
        # Proyectar sobre g(x) = 0
        alpha = -constraint_val / (jac @ jac)
        x_proj = x + alpha * jac
        
        return x_proj
    
    def _line_search_constrained(self, func: Callable, constraints: List[Callable],
                               x: np.ndarray, direction: np.ndarray) -> float:
        """Búsqueda de línea que mantiene factibilidad"""
        alpha = 1.0
        
        for _ in range(20):
            x_new = x + alpha * direction
            
            # Verificar que se mantengan las restricciones
            feasible = True
            for h in constraints:
                if abs(h(*x_new)) > 1e-6:
                    feasible = False
                    break
            
            if feasible and func(*x_new) < func(*x):
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def _line_search_feasible(self, func: Callable, constraints: List[Callable],
                            x: np.ndarray, direction: np.ndarray) -> float:
        """Búsqueda de línea que mantiene factibilidad para desigualdades"""
        alpha = 1.0
        
        # Calcular máximo step que mantiene factibilidad
        if constraints:
            for g in constraints:
                if g(*x) > 0:  # Ya viola restricción
                    return 0.0
        
        for _ in range(20):
            x_new = x + alpha * direction
            
            # Verificar factibilidad
            feasible = True
            if constraints:
                for g in constraints:
                    if g(*x_new) > 1e-6:
                        feasible = False
                        break
            
            if feasible:
                if func(*x_new) < func(*x):
                    return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def _solve_direction_subproblem(self, gradient: np.ndarray, 
                                  constraints: List[Callable],
                                  jacobians: List[Callable],
                                  x: np.ndarray) -> np.ndarray:
        """
        Resuelve el subproblema LP para encontrar dirección factible:
        min c^T d
        s.t. a_i^T d <= 0 para restricciones activas
             ||d|| <= 1
        """
        # Implementación simplificada usando proyección
        direction = -gradient / np.linalg.norm(gradient)
        
        # Proyectar sobre restricciones activas
        if constraints and jacobians:
            active_grads = []
            for g, g_jac in zip(constraints, jacobians):
                if g(*x) >= -1e-6:  # Restricción activa
                    active_grads.append(g_jac(*x))
            
            if active_grads:
                A = np.array(active_grads)
                try:
                    P = np.eye(len(x)) - A.T @ np.linalg.pinv(A @ A.T) @ A
                    direction = P @ direction
                except np.linalg.LinAlgError:
                    pass
        
        return direction
