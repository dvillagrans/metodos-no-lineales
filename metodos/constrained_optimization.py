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
    
    def projected_gradient(self, func: Callable, grad: Callable, 
                          constraints_eq: List[Callable] = None,
                          constraints_eq_jac: List[Callable] = None,
                          x0: np.ndarray = None, 
                          learning_rate: float = 0.01,
                          max_iter: int = 1000, 
                          tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de gradiente proyectado para restricciones de igualdad
        
        Args:
            func: Función objetivo
            grad: Gradiente de la función objetivo
            constraints_eq: Lista de restricciones de igualdad h(x) = 0
            constraints_eq_jac: Lista de jacobianos de las restricciones
            x0: Punto inicial
            learning_rate: Tasa de aprendizaje
            max_iter: Máximo número de iteraciones
            tol: Tolerancia para convergencia
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        violations = []
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Calcular matriz jacobiana de restricciones
            if constraints_eq and constraints_eq_jac:
                A = np.array([jac(*x) for jac in constraints_eq_jac])
                
                # Proyector ortogonal al espacio nulo de A
                if A.size > 0:
                    # P = I - A^T(AA^T)^(-1)A
                    try:
                        AAT_inv = np.linalg.pinv(A @ A.T)
                        P = np.eye(len(x)) - A.T @ AAT_inv @ A
                        projected_gradient = P @ gradient
                    except np.linalg.LinAlgError:
                        projected_gradient = gradient
                else:
                    projected_gradient = gradient
            else:
                projected_gradient = gradient
            
            # Actualizar posición
            x_new = x - learning_rate * projected_gradient
            
            # Proyectar sobre restricciones (Newton para h(x) = 0)
            if constraints_eq:
                x_new = self._project_onto_constraints(x_new, constraints_eq, constraints_eq_jac)
            
            x = x_new
            path.append(x.copy())
            errors.append(func(*x))
            
            # Calcular violación de restricciones
            violation = 0
            if constraints_eq:
                violation = sum([abs(h(*x)) for h in constraints_eq])
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
    
    def reduced_gradient(self, func: Callable, grad: Callable,
                        constraints_eq: List[Callable],
                        constraints_eq_jac: List[Callable],
                        x0: np.ndarray,
                        basic_vars: List[int] = None,
                        learning_rate: float = 0.01,
                        max_iter: int = 1000,
                        tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de gradiente reducido para restricciones de igualdad
        
        Args:
            basic_vars: Índices de variables básicas (dependientes)
        """
        x = x0.copy()
        n = len(x)
        m = len(constraints_eq)  # número de restricciones
        
        if basic_vars is None:
            basic_vars = list(range(m))  # Primeras m variables como básicas
        
        nonbasic_vars = [i for i in range(n) if i not in basic_vars]
        
        path = [x.copy()]
        errors = [func(*x)]
        violations = []
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            # Matriz jacobiana de restricciones
            A = np.array([jac(*x) for jac in constraints_eq_jac])
            
            # Dividir jacobiana en básica y no básica
            A_B = A[:, basic_vars]  # m x m
            A_N = A[:, nonbasic_vars]  # m x (n-m)
            
            # Gradiente dividido
            g_B = gradient[basic_vars]
            g_N = gradient[nonbasic_vars]
            
            try:
                # Gradiente reducido: r = g_N - A_N^T (A_B^T)^(-1) g_B
                A_B_T_inv = np.linalg.inv(A_B.T)
                reduced_grad = g_N - A_N.T @ A_B_T_inv @ g_B
                
                if np.linalg.norm(reduced_grad) < tol:
                    break
                
                # Dirección en variables no básicas
                d_N = -reduced_grad
                
                # Dirección en variables básicas: d_B = -(A_B)^(-1) A_N d_N
                A_B_inv = np.linalg.inv(A_B)
                d_B = -A_B_inv @ A_N @ d_N
                
                # Construir dirección completa
                direction = np.zeros(n)
                direction[nonbasic_vars] = d_N
                direction[basic_vars] = d_B
                
                # Búsqueda de línea
                alpha = self._line_search_constrained(func, constraints_eq, x, direction)
                x = x + alpha * direction
                
            except np.linalg.LinAlgError:
                # Si falla, usar gradiente proyectado
                A_pinv = np.linalg.pinv(A)
                P = np.eye(n) - A.T @ A_pinv.T
                direction = -P @ gradient
                x = x + learning_rate * direction
            
            path.append(x.copy())
            errors.append(func(*x))
            
            # Violación de restricciones
            violation = sum([abs(h(*x)) for h in constraints_eq])
            violations.append(violation)
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'violations': violations,
            'iterations': i + 1,
            'converged': np.linalg.norm(reduced_grad if 'reduced_grad' in locals() else gradient) < tol,
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
