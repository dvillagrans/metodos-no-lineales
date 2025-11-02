"""
Métodos Cuasi-Newton
Implementa algoritmos BFGS, L-BFGS, DFP y otros métodos cuasi-Newton
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any
from collections import deque


class QuasiNewtonOptimizer:
    """Implementa varios métodos cuasi-Newton"""
    
    def __init__(self):
        self.available_methods = [
            'bfgs',
            'lbfgs',
            'dfp',
            'sr1'
        ]
    
    def bfgs(self, func: Callable, grad: Callable, x0: np.ndarray,
             max_iter: int = 1000, tol: float = 1e-6,
             line_search: bool = True) -> Dict[str, Any]:
        """
        Algoritmo BFGS (Broyden-Fletcher-Goldfarb-Shanno)
        """
        x = x0.copy()
        n = len(x)
        H = np.eye(n)  # Aproximación inicial de la inversa de la hessiana
        
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Dirección de búsqueda
            direction = -H @ gradient
            
            # Búsqueda de línea si está habilitada
            if line_search:
                step_size = self._line_search_armijo(func, grad, x, direction)
            else:
                step_size = 1.0
            
            # Actualizar posición
            x_new = x + step_size * direction
            gradient_new = grad(*x_new)
            
            # Vectores para actualización BFGS
            s = x_new - x
            y = gradient_new - gradient
            
            # Condición de curvatura
            sy = np.dot(s, y)
            if sy > 1e-10:
                # Actualización BFGS de H
                rho = 1.0 / sy
                V = np.eye(n) - rho * np.outer(s, y)
                H = V.T @ H @ V + rho * np.outer(s, s)
            
            x = x_new
            gradient = gradient_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'bfgs'
        }
    
    def lbfgs(self, func: Callable, grad: Callable, x0: np.ndarray,
              m: int = 10, max_iter: int = 1000, tol: float = 1e-6,
              line_search: bool = True) -> Dict[str, Any]:
        """
        L-BFGS (Limited-memory BFGS)
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        # Historiales para L-BFGS
        s_history = deque(maxlen=m)
        y_history = deque(maxlen=m)
        rho_history = deque(maxlen=m)
        
        gradient = grad(*x)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Calcular dirección usando L-BFGS two-loop recursion
            direction = self._lbfgs_two_loop_recursion(gradient, s_history, y_history, rho_history)
            
            # Búsqueda de línea
            if line_search:
                step_size = self._line_search_armijo(func, grad, x, direction)
            else:
                step_size = 1.0
            
            # Actualizar posición
            x_new = x + step_size * direction
            gradient_new = grad(*x_new)
            
            # Actualizar historiales
            s = x_new - x
            y = gradient_new - gradient
            sy = np.dot(s, y)
            
            if sy > 1e-10:
                s_history.append(s)
                y_history.append(y)
                rho_history.append(1.0 / sy)
            
            x = x_new
            gradient = gradient_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'lbfgs'
        }
    
    def dfp(self, func: Callable, grad: Callable, x0: np.ndarray,
            max_iter: int = 1000, tol: float = 1e-6,
            line_search: bool = True) -> Dict[str, Any]:
        """
        Algoritmo DFP (Davidon-Fletcher-Powell)
        """
        x = x0.copy()
        n = len(x)
        H = np.eye(n)  # Aproximación inicial de la inversa de la hessiana
        
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Dirección de búsqueda
            direction = -H @ gradient
            
            # Búsqueda de línea
            if line_search:
                step_size = self._line_search_armijo(func, grad, x, direction)
            else:
                step_size = 1.0
            
            # Actualizar posición
            x_new = x + step_size * direction
            gradient_new = grad(*x_new)
            
            # Vectores para actualización DFP
            s = x_new - x
            y = gradient_new - gradient
            
            # Condición de curvatura
            sy = np.dot(s, y)
            if sy > 1e-10:
                # Actualización DFP de H
                Hy = H @ y
                yHy = np.dot(y, Hy)
                
                H = H + np.outer(s, s) / sy - np.outer(Hy, Hy) / yHy
            
            x = x_new
            gradient = gradient_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'dfp'
        }
    
    def sr1(self, func: Callable, grad: Callable, x0: np.ndarray,
            max_iter: int = 1000, tol: float = 1e-6,
            line_search: bool = True) -> Dict[str, Any]:
        """
        Método SR1 (Symmetric Rank-1)
        """
        x = x0.copy()
        n = len(x)
        H = np.eye(n)  # Aproximación inicial de la inversa de la hessiana
        
        path = [x.copy()]
        errors = [func(*x)]
        
        gradient = grad(*x)
        
        for i in range(max_iter):
            if np.linalg.norm(gradient) < tol:
                break
            
            # Dirección de búsqueda
            direction = -H @ gradient
            
            # Búsqueda de línea
            if line_search:
                step_size = self._line_search_armijo(func, grad, x, direction)
            else:
                step_size = 1.0
            
            # Actualizar posición
            x_new = x + step_size * direction
            gradient_new = grad(*x_new)
            
            # Vectores para actualización SR1
            s = x_new - x
            y = gradient_new - gradient
            
            # Actualización SR1
            u = s - H @ y
            uy = np.dot(u, y)
            
            # Evitar división por cero y mantener estabilidad
            if abs(uy) > 1e-12:
                H = H + np.outer(u, u) / uy
            
            x = x_new
            gradient = gradient_new
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'sr1'
        }
    
    def _lbfgs_two_loop_recursion(self, gradient: np.ndarray, s_history: deque, 
                                 y_history: deque, rho_history: deque) -> np.ndarray:
        """
        Implementa el algoritmo two-loop recursion para L-BFGS
        """
        q = gradient.copy()
        alpha_history = []
        
        # Primer loop (hacia atrás)
        for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alpha = rho * np.dot(s, q)
            alpha_history.append(alpha)
            q = q - alpha * y
        
        # Escalado inicial (usando la última actualización)
        if len(s_history) > 0:
            s_last = s_history[-1]
            y_last = y_history[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            r = gamma * q
        else:
            r = q
        
        alpha_history.reverse()
        
        # Segundo loop (hacia adelante)
        for i, (s, y, rho) in enumerate(zip(s_history, y_history, rho_history)):
            beta = rho * np.dot(y, r)
            r = r + (alpha_history[i] - beta) * s
        
        return -r
    
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
        
        # Condición de Armijo
        while alpha > 1e-10:
            x_new = x + alpha * direction
            f_new = func(*x_new)
            
            if f_new <= f_x + c1 * alpha * directional_derivative:
                return alpha
            
            alpha *= beta
        
        return alpha
    
    def optimize(self, method: str, func: Callable, grad: Callable, x0: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        Método unificado para llamar cualquier algoritmo cuasi-Newton
        """
        if method not in self.available_methods:
            raise ValueError(f"Método '{method}' no disponible. Métodos disponibles: {self.available_methods}")
        
        return getattr(self, method)(func, grad, x0, **kwargs)
