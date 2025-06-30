"""
Métodos de Newton
Implementa el método de Newton y sus variantes
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class NewtonOptimizer:
    """Implementa varios métodos basados en Newton"""
    
    def __init__(self):
        self.available_methods = [
            'newton_method',
            'modified_newton',
            'damped_newton',
            'gauss_newton'
        ]
    
    def newton_method(self, func: Callable, grad: Callable, hess: Callable, x0: np.ndarray,
                     max_iter: int = 100, tol: float = 1e-6,
                     regularization: float = 1e-8) -> Dict[str, Any]:
        """
        Método de Newton clásico
        Resuelve sistemas no lineales usando información de gradiente y hessiana.
        En cada iteración, se calcula la dirección de Newton resolviendo H * p = -g.
        Si la hessiana es singular o mal condicionada, se regulariza para evitar problemas numéricos.
        Si aún así falla, se recurre a un paso de gradiente descendente pequeño.
        """
        # Copiamos el punto inicial para no modificar el argumento original
        x = x0.copy()
        # Guardamos la trayectoria de puntos visitados
        path = [x.copy()]
        # Guardamos el valor de la función objetivo en cada iteración
        errors = [func(*x)]
        
        # Iteramos hasta alcanzar el máximo de iteraciones o la tolerancia
        for i in range(max_iter):
            # Calculamos el gradiente en el punto actual
            gradient = grad(*x)
            
            # Criterio de convergencia: si la norma del gradiente es pequeña, terminamos
            if np.linalg.norm(gradient) < tol:
                break
            
            # Calculamos la hessiana en el punto actual
            hessian = hess(*x)
            
            # Regularización: si la hessiana es singular o está mal condicionada, sumamos un múltiplo de la identidad
            if np.linalg.det(hessian) == 0 or np.linalg.cond(hessian) > 1e12:
                hessian += regularization * np.eye(len(x))
            
            try:
                # Resolvemos el sistema lineal H * p = -g para obtener la dirección de Newton
                direction = np.linalg.solve(hessian, -gradient)
                # Actualizamos el punto
                x = x + direction
            except np.linalg.LinAlgError:
                # Si la hessiana sigue siendo problemática, hacemos un paso pequeño de gradiente descendente
                direction = -gradient / np.linalg.norm(gradient)
                x = x + 0.01 * direction
            
            # Guardamos el nuevo punto y el error para análisis posterior
            path.append(x.copy())
            errors.append(func(*x))
        
        # Devolvemos los resultados en un diccionario estándar
        return {
            'x': x,  # Punto final encontrado
            'path': np.array(path),  # Trayectoria completa
            'errors': errors,  # Valores de la función objetivo
            'iterations': i + 1,  # Número de iteraciones realizadas
            'converged': np.linalg.norm(gradient) < tol,  # Indicador de convergencia
            'method': 'newton_method'  # Nombre del método
        }
    
    def modified_newton(self, func: Callable, grad: Callable, hess: Callable, x0: np.ndarray,
                       max_iter: int = 100, tol: float = 1e-6,
                       beta: float = 1e-3) -> Dict[str, Any]:
        """
        Método de Newton modificado con regularización adaptativa
        En cada iteración, la hessiana se modifica para asegurar que sea definida positiva,
        lo que garantiza que la dirección de búsqueda sea de descenso.
        Si la hessiana tiene autovalores pequeños o negativos, se ajustan a un mínimo beta.
        Si el sistema es singular, se recurre a un paso de gradiente descendente pequeño.
        """
        # Copiamos el punto inicial para no modificar el argumento original
        x = x0.copy()
        # Guardamos la trayectoria de puntos visitados
        path = [x.copy()]
        # Guardamos el valor de la función objetivo en cada iteración
        errors = [func(*x)]
        
        # Iteramos hasta alcanzar el máximo de iteraciones o la tolerancia
        for i in range(max_iter):
            # Calculamos el gradiente en el punto actual
            gradient = grad(*x)
            
            # Criterio de convergencia: si la norma del gradiente es pequeña, terminamos
            if np.linalg.norm(gradient) < tol:
                break
            
            # Calculamos la hessiana en el punto actual
            hessian = hess(*x)
            
            # Descomponemos la hessiana en autovalores y autovectores
            eigenvals, eigenvecs = np.linalg.eigh(hessian)
            # Ajustamos los autovalores para que sean al menos beta (evita indefinición o indefinida negativa)
            eigenvals = np.maximum(eigenvals, beta)
            # Reconstruimos la hessiana modificada
            modified_hessian = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            try:
                # Resolvemos el sistema modificado para obtener la dirección de Newton
                direction = np.linalg.solve(modified_hessian, -gradient)
                # Actualizamos el punto
                x = x + direction
            except np.linalg.LinAlgError:
                # Si la hessiana sigue siendo problemática, hacemos un paso pequeño de gradiente descendente
                direction = -gradient / np.linalg.norm(gradient)
                x = x + 0.01 * direction
            
            # Guardamos el nuevo punto y el error para análisis posterior
            path.append(x.copy())
            errors.append(func(*x))
        
        # Devolvemos los resultados en un diccionario estándar
        return {
            'x': x,  # Punto final encontrado
            'path': np.array(path),  # Trayectoria completa
            'errors': errors,  # Valores de la función objetivo
            'iterations': i + 1,  # Número de iteraciones realizadas
            'converged': np.linalg.norm(gradient) < tol,  # Indicador de convergencia
            'method': 'modified_newton'  # Nombre del método
        }
    
    def damped_newton(self, func: Callable, grad: Callable, hess: Callable, x0: np.ndarray,
                     max_iter: int = 100, tol: float = 1e-6,
                     alpha: float = 1.0, beta: float = 0.5, c1: float = 1e-4) -> Dict[str, Any]:
        """
        Método de Newton con amortiguación (damped Newton)
        """
        x = x0.copy()
        path = [x.copy()]
        errors = [func(*x)]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            hessian = hess(*x)
            
            # Regularización si es necesario
            if np.linalg.det(hessian) == 0 or np.linalg.cond(hessian) > 1e12:
                hessian += 1e-6 * np.eye(len(x))
            
            try:
                direction = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                direction = -gradient / np.linalg.norm(gradient)
            
            # Búsqueda de línea con condiciones de Armijo
            step_size = alpha
            current_func_val = func(*x)
            
            while step_size > 1e-10:
                x_new = x + step_size * direction
                new_func_val = func(*x_new)
                
                # Condición de Armijo
                if new_func_val <= current_func_val + c1 * step_size * np.dot(gradient, direction):
                    break
                
                step_size *= beta
            
            x = x + step_size * direction
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'damped_newton'
        }
    
    def gauss_newton(self, residual_func: Callable, jacobian_func: Callable, x0: np.ndarray,
                    max_iter: int = 100, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de Gauss-Newton para problemas de mínimos cuadrados
        """
        x = x0.copy()
        path = [x.copy()]
        
        def objective_func(x):
            r = residual_func(x)
            return 0.5 * np.sum(r**2)
        
        errors = [objective_func(x)]
        
        for i in range(max_iter):
            residuals = residual_func(x)
            jacobian = jacobian_func(x)
            
            # Gradiente: J^T * r
            gradient = jacobian.T @ residuals
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Aproximación de la hessiana: J^T * J
            hessian_approx = jacobian.T @ jacobian
            
            # Regularización si es necesario
            if np.linalg.det(hessian_approx) == 0 or np.linalg.cond(hessian_approx) > 1e12:
                hessian_approx += 1e-6 * np.eye(len(x))
            
            try:
                direction = np.linalg.solve(hessian_approx, -gradient)
                x = x + direction
            except np.linalg.LinAlgError:
                direction = -gradient / np.linalg.norm(gradient)
                x = x + 0.01 * direction
            
            path.append(x.copy())
            errors.append(objective_func(x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'iterations': i + 1,
            'converged': np.linalg.norm(gradient) < tol,
            'method': 'gauss_newton'
        }
    
    def numerical_hessian(self, func: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Calcular hessiana numéricamente si no está disponible"""
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: segunda derivada
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    
                    hess[i, j] = (func(*x_plus) - 2*func(*x) + func(*x_minus)) / (h**2)
                else:
                    # Fuera de diagonal: derivada mixta
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h
                    x_pp[j] += h
                    
                    x_pm[i] += h
                    x_pm[j] -= h
                    
                    x_mp[i] -= h
                    x_mp[j] += h
                    
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    hess[i, j] = (func(*x_pp) - func(*x_pm) - func(*x_mp) + func(*x_mm)) / (4 * h**2)
        
        return hess
    
    def optimize(self, method: str, func: Callable, grad: Callable, x0: np.ndarray,
                hess: Optional[Callable] = None, **kwargs) -> Dict[str, Any]:
        """
        Método unificado para llamar cualquier algoritmo de Newton
        """
        if method not in self.available_methods:
            raise ValueError(f"Método '{method}' no disponible. Métodos disponibles: {self.available_methods}")
        
        if method == 'gauss_newton':
            # Para Gauss-Newton necesitamos funciones especiales
            if 'residual_func' not in kwargs or 'jacobian_func' not in kwargs:
                raise ValueError("Para Gauss-Newton se requieren 'residual_func' y 'jacobian_func'")
            return self.gauss_newton(kwargs['residual_func'], kwargs['jacobian_func'], x0, **kwargs)
        else:
            # Para otros métodos de Newton
            if hess is None:
                # Crear hessiana numérica
                hess = lambda *args: self.numerical_hessian(func, np.array(args))
            
            return getattr(self, method)(func, grad, hess, x0, **kwargs)
