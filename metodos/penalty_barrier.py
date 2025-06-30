"""
Métodos de Penalización y Barrera
Implementa métodos de penalización exterior, interior y mixtos
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class PenaltyBarrierOptimizer:
    """Implementa métodos de penalización y barrera para optimización con restricciones"""
    
    def __init__(self):
        self.available_methods = [
            'interior_penalty_barrier',
            'logarithmic_barrier',
            'inverse_barrier',
            'mixed_penalty_barrier',
            'augmented_lagrangian'
        ]
    
    def exterior_penalty(self, func: Callable, grad: Callable,
                        constraints_eq: List[Callable] = None,
                        constraints_ineq: List[Callable] = None,
                        x0: np.ndarray = None,
                        penalty_start: float = 1.0,
                        penalty_factor: float = 10.0,
                        max_outer_iter: int = 20,
                        max_inner_iter: int = 100,
                        tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de penalización exterior
        
        Función penalizada: P(x,μ) = f(x) + μ * [∑h_i(x)² + ∑max(0,g_j(x))²]
        """
        x = x0.copy()
        mu = penalty_start
        
        path = [x.copy()]
        errors = [func(*x)]
        penalties = [mu]
        violations = []
        
        for outer_iter in range(max_outer_iter):
            # Definir función penalizada
            def penalized_func(*args):
                x_eval = np.array(args)
                f_val = func(*x_eval)
                penalty = 0.0
                
                # Penalización para restricciones de igualdad: μ * h(x)²
                if constraints_eq:
                    for h in constraints_eq:
                        penalty += h(x_eval)**2
                
                # Penalización para restricciones de desigualdad: μ * max(0, g(x))²
                if constraints_ineq:
                    for g in constraints_ineq:
                        penalty += max(0, g(x_eval))**2
                
                return f_val + mu * penalty
            
            def penalized_grad(*args):
                x_eval = np.array(args)
                grad_f = grad(*x_eval)
                grad_penalty = np.zeros_like(grad_f)
                
                # Gradiente de penalización para igualdades: 2μ * h(x) * ∇h(x)
                if constraints_eq:
                    for h in constraints_eq:
                        h_val = h(x_eval)
                        # Aproximar gradiente numéricamente
                        h_grad = self._numerical_gradient(h, x_eval)
                        grad_penalty += 2 * h_val * h_grad
                
                # Gradiente para desigualdades: 2μ * max(0,g(x)) * ∇g(x)
                if constraints_ineq:
                    for g in constraints_ineq:
                        g_val = g(x_eval)
                        if g_val > 0:
                            g_grad = self._numerical_gradient(g, x_eval)
                            grad_penalty += 2 * g_val * g_grad
                
                return grad_f + mu * grad_penalty
            
            # Optimizar función penalizada
            result = self._unconstrained_optimization(penalized_func, penalized_grad, x, max_inner_iter, tol)
            x = result['x']
            
            path.extend(result['path'][1:])  # Evitar duplicar último punto
            errors.extend([func(*xi) for xi in result['path'][1:]])
            
            # Calcular violación de restricciones
            violation = 0.0
            if constraints_eq:
                violation += sum([h(x)**2 for h in constraints_eq])
            if constraints_ineq:
                violation += sum([max(0, g(x))**2 for g in constraints_ineq])
            
            violations.append(violation)
            
            # Verificar convergencia
            if violation < tol:
                break
            
            # Aumentar parámetro de penalización
            mu *= penalty_factor
            penalties.append(mu)
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'violations': violations,
            'penalties': penalties,
            'iterations': outer_iter + 1,
            'converged': violation < tol,
            'method': 'exterior_penalty'
        }
    
    def logarithmic_barrier(self, func: Callable, grad: Callable,
                           constraints_ineq: List[Callable],
                           x0: np.ndarray,
                           barrier_start: float = 1.0,
                           barrier_factor: float = 0.1,
                           max_outer_iter: int = 20,
                           max_inner_iter: int = 100,
                           tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de barrera logarítmica
        
        Función barrera: B(x,μ) = f(x) - μ * ∑log(-g_i(x))
        Requiere que x0 sea estrictamente factible (g_i(x0) < 0)
        """
        # Verificar factibilidad inicial
        for i, g in enumerate(constraints_ineq):
            if g(x0) >= 0:
                raise ValueError(f"Punto inicial no es estrictamente factible. Restricción {i}: g(x0) = {g(x0)}")
        
        x = x0.copy()
        mu = barrier_start
        
        path = [x.copy()]
        errors = [func(*x)]
        barriers = [mu]
        violations = []
        
        for outer_iter in range(max_outer_iter):
            # Función barrera
            def barrier_func(*args):
                x_eval = np.array(args)
                f_val = func(*x_eval)
                barrier = 0.0
                
                for g in constraints_ineq:
                    g_val = g(x_eval)
                    if g_val >= 0:
                        return np.inf  # Fuera de la región factible
                    barrier -= np.log(-g_val)
                
                return f_val + mu * barrier
            
            def barrier_grad(*args):
                x_eval = np.array(args)
                grad_f = grad(*x_eval)
                grad_barrier = np.zeros_like(grad_f)
                
                for g in constraints_ineq:
                    g_val = g(x_eval)
                    if g_val >= 0:
                        return np.full_like(grad_f, np.inf)
                    
                    g_grad = self._numerical_gradient(g, x_eval)
                    grad_barrier -= g_grad / g_val
                
                return grad_f + mu * grad_barrier
            
            # Optimizar función barrera
            result = self._unconstrained_optimization(barrier_func, barrier_grad, x, max_inner_iter, tol)
            x = result['x']
            
            path.extend(result['path'][1:])
            errors.extend([func(*xi) for xi in result['path'][1:]])
            
            # Calcular violación (distancia a la frontera)
            violation = min([-g(x) for g in constraints_ineq])
            violations.append(violation)
            
            # Verificar convergencia
            if mu < tol:
                break
            
            # Reducir parámetro de barrera
            mu *= barrier_factor
            barriers.append(mu)
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'violations': violations,
            'barriers': barriers,
            'iterations': outer_iter + 1,
            'converged': mu < tol,
            'method': 'logarithmic_barrier'
        }
    
    def inverse_barrier(self, func: Callable, grad: Callable,
                       constraints_ineq: List[Callable],
                       x0: np.ndarray,
                       barrier_start: float = 1.0,
                       barrier_factor: float = 0.1,
                       max_outer_iter: int = 20,
                       max_inner_iter: int = 100,
                       tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de barrera inversa
        
        Función barrera: B(x,μ) = f(x) + μ * ∑(1/(-g_i(x)))
        """
        # Verificar factibilidad inicial
        for i, g in enumerate(constraints_ineq):
            if g(x0) >= 0:
                raise ValueError(f"Punto inicial no es estrictamente factible. Restricción {i}: g(x0) = {g(x0)}")
        
        x = x0.copy()
        mu = barrier_start
        
        path = [x.copy()]
        errors = [func(*x)]
        barriers = [mu]
        
        for outer_iter in range(max_outer_iter):
            # Función barrera inversa
            def barrier_func(*args):
                x_eval = np.array(args)
                f_val = func(*x_eval)
                barrier = 0.0
                
                for g in constraints_ineq:
                    g_val = g(*x_eval)
                    if g_val >= 0:
                        return np.inf
                    barrier += 1.0 / (-g_val)
                
                return f_val + mu * barrier
            
            def barrier_grad(*args):
                x_eval = np.array(args)
                grad_f = grad(*x_eval)
                grad_barrier = np.zeros_like(grad_f)
                
                for g in constraints_ineq:
                    g_val = g(*x_eval)
                    if g_val >= 0:
                        return np.full_like(grad_f, np.inf)
                    
                    g_grad = self._numerical_gradient(g, x_eval)
                    grad_barrier += g_grad / (g_val**2)
                
                return grad_f + mu * grad_barrier
            
            # Optimizar
            result = self._unconstrained_optimization(barrier_func, barrier_grad, x, max_inner_iter, tol)
            x = result['x']
            
            path.extend(result['path'][1:])
            errors.extend([func(*xi) for xi in result['path'][1:]])
            
            # Verificar convergencia
            if mu < tol:
                break
            
            mu *= barrier_factor
            barriers.append(mu)
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'barriers': barriers,
            'iterations': outer_iter + 1,
            'converged': mu < tol,
            'method': 'inverse_barrier'
        }
    
    def mixed_penalty_barrier(self, func: Callable, grad: Callable,
                             constraints_eq: List[Callable] = None,
                             constraints_ineq: List[Callable] = None,
                             x0: np.ndarray = None,
                             penalty_start: float = 1.0,
                             barrier_start: float = 1.0,
                             penalty_factor: float = 10.0,
                             barrier_factor: float = 0.1,
                             max_outer_iter: int = 20,
                             max_inner_iter: int = 100,
                             tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método mixto de penalización (igualdades) y barrera (desigualdades)
        """
        x = x0.copy()
        mu_p = penalty_start  # Para penalización
        mu_b = barrier_start  # Para barrera
        
        path = [x.copy()]
        errors = [func(*x)]
        
        for outer_iter in range(max_outer_iter):
            # Función mixta
            def mixed_func(*args):
                x_eval = np.array(args)
                f_val = func(*x_eval)
                penalty = 0.0
                barrier = 0.0
                
                # Penalización para igualdades
                if constraints_eq:
                    for h in constraints_eq:
                        penalty += h(*x_eval)**2
                
                # Barrera para desigualdades
                if constraints_ineq:
                    for g in constraints_ineq:
                        g_val = g(*x_eval)
                        if g_val >= 0:
                            return np.inf
                        barrier -= np.log(-g_val)
                
                return f_val + mu_p * penalty + mu_b * barrier
            
            def mixed_grad(*args):
                x_eval = np.array(args)
                grad_f = grad(*x_eval)
                grad_total = grad_f.copy()
                
                # Gradiente de penalización
                if constraints_eq:
                    for h in constraints_eq:
                        h_val = h(*x_eval)
                        h_grad = self._numerical_gradient(h, x_eval)
                        grad_total += 2 * mu_p * h_val * h_grad
                
                # Gradiente de barrera
                if constraints_ineq:
                    for g in constraints_ineq:
                        g_val = g(*x_eval)
                        if g_val >= 0:
                            return np.full_like(grad_f, np.inf)
                        g_grad = self._numerical_gradient(g, x_eval)
                        grad_total -= mu_b * g_grad / g_val
                
                return grad_total
            
            # Optimizar función mixta
            result = self._unconstrained_optimization(mixed_func, mixed_grad, x, max_inner_iter, tol)
            x = result['x']
            
            path.extend(result['path'][1:])
            errors.extend([func(*xi) for xi in result['path'][1:]])
            
            # Verificar convergencia
            violation_eq = 0.0
            if constraints_eq:
                violation_eq = sum([h(x)**2 for h in constraints_eq])
            
            if violation_eq < tol and mu_b < tol:
                break
            
            # Actualizar parámetros
            mu_p *= penalty_factor
            mu_b *= barrier_factor
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': outer_iter + 1,
            'converged': violation_eq < tol and mu_b < tol,
            'method': 'mixed_penalty_barrier'
        }
    
    def _numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Calcula el gradiente numérico de una función"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            # Intentar con argumentos desempaquetados primero, luego con array
            try:
                grad[i] = (func(*x_plus) - func(*x_minus)) / (2 * h)
            except TypeError:
                # Si falla, la función espera un array
                grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad
    
    def _unconstrained_optimization(self, func: Callable, grad: Callable, x0: np.ndarray,
                                  max_iter: int, tol: float) -> Dict[str, Any]:
        """Optimización sin restricciones usando gradiente descendente"""
        x = x0.copy()
        path = [x.copy()]
        learning_rate = 0.01
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Búsqueda de línea simple
            alpha = learning_rate
            for _ in range(10):
                x_new = x - alpha * gradient
                if func(*x_new) < func(*x):
                    break
                alpha *= 0.5
            
            x = x - alpha * gradient
            path.append(x.copy())
        
        return {
            'x': x,
            'path': path,
            'iterations': i + 1
        }
