"""
Métodos de Búsqueda de Línea
Implementa diferentes algoritmos de búsqueda de línea (line search)
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class LineSearchOptimizer:
    """Implementa varios métodos de búsqueda de línea"""
    
    def __init__(self):
        self.available_methods = [
            'armijo_backtracking',
            'wolfe_conditions',
            'golden_section',
            'fibonacci_search',
            'quadratic_interpolation',
            'cubic_interpolation'
        ]
    
    def armijo_backtracking(self, func: Callable, grad: Callable, x0: np.ndarray,
                           direction: np.ndarray, c1: float = 1e-4, rho: float = 0.5,
                           alpha_max: float = 1.0, max_iter: int = 1000,
                           tol: float = 1e-6) -> Dict[str, Any]:
        """
        Búsqueda de línea con backtracking usando condiciones de Armijo
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Usar dirección proporcionada o gradiente descendente
            if direction is None:
                search_direction = -gradient
            else:
                search_direction = direction
            
            # Búsqueda de línea
            alpha = self._armijo_line_search(func, grad, x, search_direction, c1, rho, alpha_max)
            
            # Actualizar posición
            x = x + alpha * search_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'armijo_backtracking'
        }
    
    def wolfe_conditions(self, func: Callable, grad: Callable, x0: np.ndarray,
                        direction: np.ndarray = None, c1: float = 1e-4, c2: float = 0.9,
                        alpha_max: float = 1.0, max_iter: int = 1000,
                        tol: float = 1e-6) -> Dict[str, Any]:
        """
        Búsqueda de línea usando condiciones de Wolfe
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Usar dirección proporcionada o gradiente descendente
            if direction is None:
                search_direction = -gradient
            else:
                search_direction = direction
            
            # Búsqueda de línea con condiciones de Wolfe
            alpha = self._wolfe_line_search(func, grad, x, search_direction, c1, c2, alpha_max)
            
            # Actualizar posición
            x = x + alpha * search_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'wolfe_conditions'
        }
    
    def golden_section(self, func: Callable, grad: Callable, x0: np.ndarray,
                      direction: np.ndarray = None, a: float = 0.0, b: float = 1.0,
                      max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Búsqueda de línea usando sección áurea
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Usar dirección proporcionada o gradiente descendente
            if direction is None:
                search_direction = -gradient
            else:
                search_direction = direction
            
            # Función unidimensional para la búsqueda de línea
            def phi(alpha):
                return func(*(x + alpha * search_direction))
            
            # Búsqueda de línea usando sección áurea
            alpha = self._golden_section_search(phi, a, b, tol)
            
            # Actualizar posición
            x = x + alpha * search_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'golden_section'
        }
    
    def fibonacci_search(self, func: Callable, grad: Callable, x0: np.ndarray,
                        direction: np.ndarray = None, a: float = 0.0, b: float = 1.0,
                        n: int = 20, max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Búsqueda de línea usando búsqueda de Fibonacci
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Usar dirección proporcionada o gradiente descendente
            if direction is None:
                search_direction = -gradient
            else:
                search_direction = direction
            
            # Función unidimensional para la búsqueda de línea
            def phi(alpha):
                return func(*(x + alpha * search_direction))
            
            # Búsqueda de línea usando Fibonacci
            alpha = self._fibonacci_search(phi, a, b, n)
            
            # Actualizar posición
            x = x + alpha * search_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'fibonacci_search'
        }
    
    def quadratic_interpolation(self, func: Callable, grad: Callable, x0: np.ndarray,
                               direction: np.ndarray = None, alpha_max: float = 1.0,
                               max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Búsqueda de línea usando interpolación cuadrática
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Usar dirección proporcionada o gradiente descendente
            if direction is None:
                search_direction = -gradient
            else:
                search_direction = direction
            
            # Función unidimensional para la búsqueda de línea
            def phi(alpha):
                return func(*(x + alpha * search_direction))
            
            # Búsqueda de línea usando interpolación cuadrática
            alpha = self._quadratic_interpolation_search(phi, alpha_max)
            
            # Actualizar posición
            x = x + alpha * search_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'quadratic_interpolation'
        }
    
    def cubic_interpolation(self, func: Callable, grad: Callable, x0: np.ndarray,
                           direction: np.ndarray = None, alpha_max: float = 1.0,
                           max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Búsqueda de línea usando interpolación cúbica
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Usar dirección proporcionada o gradiente descendente
            if direction is None:
                search_direction = -gradient
            else:
                search_direction = direction
            
            # Función unidimensional y su derivada
            def phi(alpha):
                return func(*(x + alpha * search_direction))
            
            def phi_prime(alpha):
                x_alpha = x + alpha * search_direction
                grad_alpha = grad(*x_alpha)
                return np.dot(grad_alpha, search_direction)
            
            # Búsqueda de línea usando interpolación cúbica
            alpha = self._cubic_interpolation_search(phi, phi_prime, alpha_max)
            
            # Actualizar posición
            x = x + alpha * search_direction
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'cubic_interpolation'
        }
    
    # Métodos auxiliares para búsqueda de línea
    
    def _armijo_line_search(self, func: Callable, grad: Callable, x: np.ndarray,
                           direction: np.ndarray, c1: float, rho: float,
                           alpha_max: float) -> float:
        """Implementa búsqueda de línea con condiciones de Armijo"""
        alpha = alpha_max
        f_x = func(*x)
        grad_x = grad(*x)
        directional_derivative = np.dot(grad_x, direction)
        
        while alpha > 1e-16:
            x_new = x + alpha * direction
            f_new = func(*x_new)
            
            if f_new <= f_x + c1 * alpha * directional_derivative:
                return alpha
            
            alpha *= rho
        
        return alpha
    
    def _wolfe_line_search(self, func: Callable, grad: Callable, x: np.ndarray,
                          direction: np.ndarray, c1: float, c2: float,
                          alpha_max: float) -> float:
        """Implementa búsqueda de línea con condiciones de Wolfe"""
        alpha = alpha_max
        f_x = func(*x)
        grad_x = grad(*x)
        directional_derivative = np.dot(grad_x, direction)
        
        for _ in range(50):  # Máximo de iteraciones
            x_new = x + alpha * direction
            f_new = func(*x_new)
            grad_new = grad(*x_new)
            new_directional_derivative = np.dot(grad_new, direction)
            
            # Condición de Armijo
            if f_new > f_x + c1 * alpha * directional_derivative:
                alpha *= 0.5
                continue
            
            # Condición de curvatura
            if abs(new_directional_derivative) <= c2 * abs(directional_derivative):
                return alpha
            
            if new_directional_derivative >= 0:
                alpha *= 0.5
            else:
                alpha *= 2.0
        
        return alpha
    
    def _golden_section_search(self, func: Callable, a: float, b: float, tol: float) -> float:
        """Implementa búsqueda de sección áurea"""
        phi = (1 + np.sqrt(5)) / 2  # Número áureo
        resphi = 2 - phi
        
        # Inicializar puntos
        x1 = a + resphi * (b - a)
        x2 = a + (1 - resphi) * (b - a)
        f1 = func(x1)
        f2 = func(x2)
        
        while abs(b - a) > tol:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = func(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (1 - resphi) * (b - a)
                f2 = func(x2)
        
        return (a + b) / 2
    
    def _fibonacci_search(self, func: Callable, a: float, b: float, n: int) -> float:
        """Implementa búsqueda de Fibonacci"""
        # Generar números de Fibonacci
        fib = [1, 1]
        for i in range(2, n + 1):
            fib.append(fib[i-1] + fib[i-2])
        
        # Inicializar puntos
        x1 = a + (fib[n-2] / fib[n]) * (b - a)
        x2 = a + (fib[n-1] / fib[n]) * (b - a)
        f1 = func(x1)
        f2 = func(x2)
        
        for k in range(n - 1):
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + (fib[n-k-3] / fib[n-k-1]) * (b - a)
                f1 = func(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (fib[n-k-2] / fib[n-k-1]) * (b - a)
                f2 = func(x2)
        
        return (a + b) / 2
    
    def _quadratic_interpolation_search(self, func: Callable, alpha_max: float) -> float:
        """Implementa búsqueda usando interpolación cuadrática"""
        # Evaluar en tres puntos
        alpha0 = 0.0
        alpha1 = alpha_max / 2
        alpha2 = alpha_max
        
        f0 = func(alpha0)
        f1 = func(alpha1)
        f2 = func(alpha2)
        
        # Calcular coeficientes del polinomio cuadrático
        # f(α) = a*α² + b*α + c
        denominator = (alpha0 - alpha1) * (alpha0 - alpha2) * (alpha1 - alpha2)
        
        if abs(denominator) < 1e-12:
            return alpha1
        
        a = (alpha2 * (f1 - f0) + alpha1 * (f0 - f2) + alpha0 * (f2 - f1)) / denominator
        b = (alpha2**2 * (f0 - f1) + alpha1**2 * (f2 - f0) + alpha0**2 * (f1 - f2)) / denominator
        
        # Mínimo del polinomio cuadrático
        if abs(a) < 1e-12:
            return alpha1
        
        alpha_min = -b / (2 * a)
        
        # Verificar que esté en el rango válido
        if alpha_min < 0 or alpha_min > alpha_max:
            return alpha1
        
        return alpha_min
    
    def _cubic_interpolation_search(self, func: Callable, func_prime: Callable, alpha_max: float) -> float:
        """Implementa búsqueda usando interpolación cúbica"""
        # Evaluar en dos puntos
        alpha0 = 0.0
        alpha1 = alpha_max
        
        f0 = func(alpha0)
        f1 = func(alpha1)
        fp0 = func_prime(alpha0)
        fp1 = func_prime(alpha1)
        
        # Calcular coeficientes del polinomio cúbico
        # f(α) = a*α³ + b*α² + c*α + d
        d = f0
        c = fp0
        
        denominator = alpha1**3
        if abs(denominator) < 1e-12:
            return alpha1 / 2
        
        a = (fp1 - fp0 - 2 * (f1 - f0 - fp0 * alpha1) / alpha1) / denominator
        b = (f1 - f0 - fp0 * alpha1 - a * alpha1**3) / (alpha1**2)
        
        # Encontrar el mínimo resolviendo f'(α) = 0
        # f'(α) = 3*a*α² + 2*b*α + c = 0
        discriminant = 4 * b**2 - 12 * a * c
        
        if discriminant < 0 or abs(a) < 1e-12:
            return alpha1 / 2
        
        alpha_min1 = (-2 * b + np.sqrt(discriminant)) / (6 * a)
        alpha_min2 = (-2 * b - np.sqrt(discriminant)) / (6 * a)
        
        # Elegir el mínimo que esté en el rango válido
        candidates = [alpha for alpha in [alpha_min1, alpha_min2] if 0 <= alpha <= alpha_max]
        
        if not candidates:
            return alpha1 / 2
        
        # Evaluar candidatos y elegir el mejor
        best_alpha = candidates[0]
        best_value = func(best_alpha)
        
        for alpha in candidates[1:]:
            value = func(alpha)
            if value < best_value:
                best_alpha = alpha
                best_value = value
        
        return best_alpha
    
    def optimize(self, method: str, func: Callable, grad: Callable, x0: np.ndarray,
                direction: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Método unificado para llamar cualquier algoritmo de búsqueda de línea
        """
        if method not in self.available_methods:
            raise ValueError(f"Método '{method}' no disponible. Métodos disponibles: {self.available_methods}")
        
        return getattr(self, method)(func, grad, x0, direction, **kwargs)
