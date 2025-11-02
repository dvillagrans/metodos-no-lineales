"""
Métodos de Gradientes Conjugados
Implementa varios algoritmos de gradientes conjugados
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class ConjugateGradientOptimizer:
    """Implementa varios métodos de gradientes conjugados"""
    
    def __init__(self):
        self.available_methods = [
            'conjugate_gradient_fr',  # Fletcher-Reeves
            'conjugate_gradient_pr',  # Polak-Ribière
            'conjugate_gradient_hs',  # Hestenes-Stiefel
            'conjugate_gradient_dy',  # Dai-Yuan
            'conjugate_gradient_linear'  # Para sistemas lineales
        ]
    
    def conjugate_gradient_fr(self, func: Callable, grad: Callable, x0: np.ndarray,
                             max_iter: int = 1000, tol: float = 1e-6,
                             restart_iter: int = None) -> Dict[str, Any]:
        """
        Gradientes conjugados Fletcher-Reeves
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        direction = -gradient.copy()
        
        if restart_iter is None:
            restart_iter = len(x0)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Búsqueda de línea
            alpha = self._line_search_armijo(func, grad, x, direction)
            
            # Actualizar posición
            x_new = x + alpha * direction
            gradient_new = grad(*x_new)
            
            # Calcular beta usando Fletcher-Reeves
            beta_fr = np.dot(gradient_new, gradient_new) / np.dot(gradient, gradient)
            
            # Actualizar dirección
            if (i + 1) % restart_iter == 0:
                # Reiniciar cada restart_iter iteraciones
                direction_new = -gradient_new
            else:
                direction_new = -gradient_new + beta_fr * direction
            
            # Actualizar variables
            x = x_new
            gradient = gradient_new
            direction = direction_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'conjugate_gradient_fr'
        }
    
    def conjugate_gradient_pr(self, func: Callable, grad: Callable, x0: np.ndarray,
                             max_iter: int = 1000, tol: float = 1e-6,
                             restart_iter: int = None) -> Dict[str, Any]:
        """
        Gradientes conjugados Polak-Ribière
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        direction = -gradient.copy()
        
        if restart_iter is None:
            restart_iter = len(x0)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Búsqueda de línea
            alpha = self._line_search_armijo(func, grad, x, direction)
            
            # Actualizar posición
            x_new = x + alpha * direction
            gradient_new = grad(*x_new)
            
            # Calcular beta usando Polak-Ribière
            y = gradient_new - gradient
            beta_pr = np.dot(gradient_new, y) / np.dot(gradient, gradient)
            beta_pr = max(0, beta_pr)  # Asegurar que beta >= 0
            
            # Actualizar dirección
            if (i + 1) % restart_iter == 0:
                # Reiniciar cada restart_iter iteraciones
                direction_new = -gradient_new
            else:
                direction_new = -gradient_new + beta_pr * direction
            
            # Actualizar variables
            x = x_new
            gradient = gradient_new
            direction = direction_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'conjugate_gradient_pr'
        }
    
    def conjugate_gradient_hs(self, func: Callable, grad: Callable, x0: np.ndarray,
                             max_iter: int = 1000, tol: float = 1e-6,
                             restart_iter: int = None) -> Dict[str, Any]:
        """
        Gradientes conjugados Hestenes-Stiefel
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        direction = -gradient.copy()
        
        if restart_iter is None:
            restart_iter = len(x0)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Búsqueda de línea
            alpha = self._line_search_armijo(func, grad, x, direction)
            
            # Actualizar posición
            x_new = x + alpha * direction
            gradient_new = grad(*x_new)
            
            # Calcular beta usando Hestenes-Stiefel
            y = gradient_new - gradient
            denominator = np.dot(direction, y)
            
            if abs(denominator) > 1e-12:
                beta_hs = np.dot(gradient_new, y) / denominator
            else:
                beta_hs = 0.0
            
            # Actualizar dirección
            if (i + 1) % restart_iter == 0:
                # Reiniciar cada restart_iter iteraciones
                direction_new = -gradient_new
            else:
                direction_new = -gradient_new + beta_hs * direction
            
            # Actualizar variables
            x = x_new
            gradient = gradient_new
            direction = direction_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'conjugate_gradient_hs'
        }
    
    def conjugate_gradient_dy(self, func: Callable, grad: Callable, x0: np.ndarray,
                             max_iter: int = 1000, tol: float = 1e-6,
                             restart_iter: int = None) -> Dict[str, Any]:
        """
        Gradientes conjugados Dai-Yuan
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        direction = -gradient.copy()
        
        if restart_iter is None:
            restart_iter = len(x0)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Búsqueda de línea
            alpha = self._line_search_armijo(func, grad, x, direction)
            
            # Actualizar posición
            x_new = x + alpha * direction
            gradient_new = grad(*x_new)
            
            # Calcular beta usando Dai-Yuan
            y = gradient_new - gradient
            denominator = np.dot(direction, y)
            
            if abs(denominator) > 1e-12:
                beta_dy = np.dot(gradient_new, gradient_new) / denominator
            else:
                beta_dy = 0.0
            
            # Actualizar dirección
            if (i + 1) % restart_iter == 0:
                # Reiniciar cada restart_iter iteraciones
                direction_new = -gradient_new
            else:
                direction_new = -gradient_new + beta_dy * direction
            
            # Actualizar variables
            x = x_new
            gradient = gradient_new
            direction = direction_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'conjugate_gradient_dy'
        }
    
    def conjugate_gradient_linear(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray = None,
                                 max_iter: int = None, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Gradientes conjugados para sistemas lineales Ax = b
        """
        n = len(b)
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        if max_iter is None:
            max_iter = n
        
        path = [x.copy()]
        
        # Residuo inicial
        r = b - A @ x
        direction = r.copy()
        
        errors = [np.linalg.norm(r)]
        
        for i in range(max_iter):
            if np.linalg.norm(r) < tol:
                break
            
            # Calcular alpha
            Ad = A @ direction
            alpha = np.dot(r, r) / np.dot(direction, Ad)
            
            # Actualizar solución
            x = x + alpha * direction
            
            # Actualizar residuo
            r_new = r - alpha * Ad
            
            # Calcular beta
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            
            # Actualizar dirección
            direction = r_new + beta * direction
            
            # Actualizar residuo
            r = r_new
            
            path.append(x.copy())
            errors.append(np.linalg.norm(r))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(r) < tol,
            'method': 'conjugate_gradient_linear'
        }
    
    def preconditioned_conjugate_gradient(self, A: np.ndarray, b: np.ndarray, 
                                         M: np.ndarray = None, x0: np.ndarray = None,
                                         max_iter: int = None, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Gradientes conjugados precondicionados para sistemas lineales
        """
        n = len(b)
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        if max_iter is None:
            max_iter = n
        
        if M is None:
            M = np.eye(n)  # Sin precondicionamiento
        
        path = [x.copy()]
        
        # Residuo inicial
        r = b - A @ x
        
        # Resolver M * z = r
        z = np.linalg.solve(M, r)
        direction = z.copy()
        
        errors = [np.linalg.norm(r)]
        
        for i in range(max_iter):
            if np.linalg.norm(r) < tol:
                break
            
            # Calcular alpha
            Ad = A @ direction
            alpha = np.dot(r, z) / np.dot(direction, Ad)
            
            # Actualizar solución
            x = x + alpha * direction
            
            # Actualizar residuo
            r_new = r - alpha * Ad
            
            # Resolver M * z_new = r_new
            z_new = np.linalg.solve(M, r_new)
            
            # Calcular beta
            beta = np.dot(r_new, z_new) / np.dot(r, z)
            
            # Actualizar dirección
            direction = z_new + beta * direction
            
            # Actualizar variables
            r = r_new
            z = z_new
            
            path.append(x.copy())
            errors.append(np.linalg.norm(r))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(r) < tol,
            'method': 'preconditioned_conjugate_gradient'
        }
    
    def _line_search_armijo(self, func: Callable, grad: Callable, x: np.ndarray,
                           direction: np.ndarray, c1: float = 1e-4,
                           alpha_init: float = 1.0, beta: float = 0.5) -> float:
        """
        Búsqueda de línea usando condiciones de Armijo
        """
        alpha = alpha_init
        f_x = func(*x)
        grad_x = grad(*x)
        directional_derivative = np.dot(grad_x, direction)
        
        # Asegurar que la dirección sea de descenso
        if directional_derivative >= 0:
            return 0.01  # Paso pequeño si no es dirección de descenso
        
        while alpha > 1e-16:
            x_new = x + alpha * direction
            f_new = func(*x_new)
            
            if f_new <= f_x + c1 * alpha * directional_derivative:
                return alpha
            
            alpha *= beta
        
        return alpha
    
    def _exact_line_search_quadratic(self, func: Callable, grad: Callable, 
                                    x: np.ndarray, direction: np.ndarray) -> float:
        """
        Búsqueda de línea exacta para funciones cuadráticas
        Útil cuando se conoce que la función es cuadrática
        """
        grad_x = grad(*x)
        
        # Para función cuadrática f(x) = 0.5 * x^T * A * x - b^T * x + c
        # El paso óptimo es: alpha = -(g^T * d) / (d^T * A * d)
        
        # Aproximar A * d usando diferencias finitas
        h = 1e-8
        x_plus = x + h * direction
        grad_plus = grad(*x_plus)
        A_times_d = (grad_plus - grad_x) / h
        
        denominator = np.dot(direction, A_times_d)
        numerator = -np.dot(grad_x, direction)
        
        if abs(denominator) > 1e-12:
            alpha = numerator / denominator
            return max(0.0, alpha)  # Asegurar que alpha >= 0
        else:
            # Fallback a búsqueda de línea estándar
            return self._line_search_armijo(func, grad, x, direction)
    
    def optimize(self, method: str, func: Callable = None, grad: Callable = None, 
                x0: np.ndarray = None, A: np.ndarray = None, b: np.ndarray = None,
                **kwargs) -> Dict[str, Any]:
        """
        Método unificado para llamar cualquier algoritmo de gradientes conjugados
        """
        if method not in self.available_methods:
            raise ValueError(f"Método '{method}' no disponible. Métodos disponibles: {self.available_methods}")
        
        if method == 'conjugate_gradient_linear':
            if A is None or b is None:
                raise ValueError("Para 'conjugate_gradient_linear' se requieren matrices A y b")
            return self.conjugate_gradient_linear(A, b, x0, **kwargs)
        else:
            if func is None or grad is None or x0 is None:
                raise ValueError(f"Para '{method}' se requieren func, grad y x0")
            return getattr(self, method)(func, grad, x0, **kwargs)
