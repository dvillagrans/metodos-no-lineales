"""
Métodos de Gradiente Descendente
Implementa varios algoritmos basados en gradiente descendente
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class GradientDescentOptimizer:
    """Implementa varios métodos de gradiente descendente"""
    
    def __init__(self):
        self.available_methods = [
            'gradient_descent',
            'gradient_descent_momentum',
            'gradient_descent_adaptive',
            'rmsprop',
            'adam'
        ]
    
    def gradient_descent(self, func: Callable, grad: Callable, x0: np.ndarray, 
                        learning_rate: float = 0.01, max_iter: int = 1000, 
                        tol: float = 1e-6) -> Dict[str, Any]:
        """
        Gradiente descendente básico
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
                
            x = x - learning_rate * gradient
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'gradient_descent'
        }
    
    def gradient_descent_momentum(self, func: Callable, grad: Callable, x0: np.ndarray,
                                 learning_rate: float = 0.01, momentum: float = 0.9,
                                 max_iter: int = 1000, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Gradiente descendente con momentum
        """
        x = x0.copy()
        velocity = np.zeros_like(x)
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            velocity = momentum * velocity - learning_rate * gradient
            x = x + velocity
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'gradient_descent_momentum'
        }
    
    def gradient_descent_adaptive(self, func: Callable, grad: Callable, x0: np.ndarray,
                                 learning_rate: float = 0.1, max_iter: int = 1000,
                                 tol: float = 1e-6, decay_factor: float = 0.9) -> Dict[str, Any]:
        """
        Gradiente descendente con tasa de aprendizaje adaptativa
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        current_lr = learning_rate
        prev_error = func(*x)
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            x_new = x - current_lr * gradient
            new_error = func(*x_new)
            
            # Si el error aumenta, reducir la tasa de aprendizaje
            if new_error > prev_error:
                current_lr *= decay_factor
                x_new = x - current_lr * gradient
                new_error = func(*x_new)
            
            x = x_new
            prev_error = new_error
            
            path.append(x.copy())
            errors.append(new_error)
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'gradient_descent_adaptive'
        }
    
    def rmsprop(self, func: Callable, grad: Callable, x0: np.ndarray,
                learning_rate: float = 0.01, decay_rate: float = 0.9,
                epsilon: float = 1e-8, max_iter: int = 1000,
                tol: float = 1e-6) -> Dict[str, Any]:
        """
        RMSprop optimizer
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        v = np.zeros_like(x)
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            v = decay_rate * v + (1 - decay_rate) * gradient**2
            x = x - learning_rate * gradient / (np.sqrt(v) + epsilon)
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'rmsprop'
        }
    
    def adam(self, func: Callable, grad: Callable, x0: np.ndarray,
             learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
             epsilon: float = 1e-8, max_iter: int = 1000,
             tol: float = 1e-6) -> Dict[str, Any]:
        """
        Adam optimizer
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * gradient**2
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1**(i + 1))
            
            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - beta2**(i + 1))
            
            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'adam'
        }
    
    def optimize(self, method: str, func: Callable, grad: Callable, x0: np.ndarray,
                **kwargs) -> Dict[str, Any]:
        """
        Método unificado para llamar cualquier algoritmo de gradiente descendente
        """
        if method not in self.available_methods:
            raise ValueError(f"Método '{method}' no disponible. Métodos disponibles: {self.available_methods}")
        
        return getattr(self, method)(func, grad, x0, **kwargs)
