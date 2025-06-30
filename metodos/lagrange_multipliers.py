"""
Método de Multiplicadores de Lagrange
Implementa métodos basados en la función lagrangiana para optimización con restricciones
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


class LagrangeMultiplierOptimizer:
    """Implementa métodos de multiplicadores de Lagrange y sus variantes"""
    
    def __init__(self):
        self.available_methods = [
            'lagrange_newton',
            'augmented_lagrangian',
            'sequential_quadratic_programming',
            'penalty_lagrangian'
        ]
    
    def lagrange_newton(self, func: Callable, grad: Callable, hess: Callable,
                       constraints_eq: List[Callable],
                       constraints_eq_jac: List[Callable],
                       constraints_eq_hess: List[Callable] = None,
                       x0: np.ndarray = None,
                       lambda0: np.ndarray = None,
                       max_iter: int = 100,
                       tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método de Newton para las condiciones KKT del Lagrangiano
        
        L(x,λ) = f(x) + λᵀh(x)
        
        Condiciones KKT:
        ∇ₓL = ∇f(x) + Σλᵢ∇hᵢ(x) = 0
        h(x) = 0
        """
        n = len(x0)
        m = len(constraints_eq)
        
        x = x0.copy()
        if lambda0 is None:
            lam = np.zeros(m)
        else:
            lam = lambda0.copy()
        
        path_x = [x.copy()]
        path_lambda = [lam.copy()]
        errors = [func(*x)]
        violations = []
        
        for iteration in range(max_iter):
            # Evaluar funciones y derivadas
            f_grad = grad(*x)
            f_hess = hess(*x)
            
            h_vals = np.array([h(x) for h in constraints_eq])
            h_jacs = np.array([jac(x) for jac in constraints_eq_jac])
            
            # Gradiente del Lagrangiano respecto a x
            grad_L_x = f_grad + h_jacs.T @ lam
            
            # Calcular hessiana del Lagrangiano
            if constraints_eq_hess:
                hess_L_x = f_hess + sum([lam[i] * hess_h(x) for i, hess_h in enumerate(constraints_eq_hess)])
            else:
                # Aproximación cuasi-Newton (solo hessiana de f)
                hess_L_x = f_hess
            
            # Construir sistema KKT:
            # [H_L   A^T] [Δx] = -[∇_x L]
            # [A     0  ] [Δλ]   -[h(x)]
            
            KKT_matrix = np.block([
                [hess_L_x, h_jacs.T],
                [h_jacs, np.zeros((m, m))]
            ])
            
            rhs = -np.concatenate([grad_L_x, h_vals])
            
            # Calcular violación de las condiciones KKT
            violation = np.linalg.norm(rhs)
            violations.append(violation)
            
            if violation < tol:
                break
            
            try:
                # Resolver sistema KKT
                delta = np.linalg.solve(KKT_matrix, rhs)
                delta_x = delta[:n]
                delta_lambda = delta[n:]
                
                # Actualizar variables
                alpha = self._line_search_lagrangian(func, constraints_eq, x, lam, delta_x, delta_lambda)
                x = x + alpha * delta_x
                lam = lam + alpha * delta_lambda
                
            except np.linalg.LinAlgError:
                # Si falla, usar paso pequeño en dirección del gradiente
                alpha = 0.01
                x = x - alpha * grad_L_x
                # Actualizar multiplicadores usando mínimos cuadrados
                try:
                    lam = np.linalg.pinv(h_jacs) @ (-f_grad)
                except:
                    pass
            
            path_x.append(x.copy())
            path_lambda.append(lam.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'lambda': lam,
            'path': np.array(path_x),  # Usar path estándar
            'path_lambda': np.array(path_lambda),
            'errors': errors,
            'violations': violations,
            'iterations': iteration + 1,
            'converged': violation < tol,
            'method': 'lagrange_newton'
        }
    
    def augmented_lagrangian(self, func: Callable, grad: Callable,
                            constraints_eq: List[Callable] = None,
                            constraints_eq_jac: List[Callable] = None,
                            constraints_ineq: List[Callable] = None,
                            constraints_ineq_jac: List[Callable] = None,
                            x0: np.ndarray = None,
                            lambda_eq0: np.ndarray = None,
                            lambda_ineq0: np.ndarray = None,
                            penalty_start: float = 1.0,
                            penalty_factor: float = 10.0,
                            max_outer_iter: int = 20,
                            max_inner_iter: int = 100,
                            tol: float = 1e-6) -> Dict[str, Any]:
        """
        Método del Lagrangiano Aumentado (Método de Hestenes-Powell)
        
        LA(x,λ,μ) = f(x) + λᵀh(x) + (μ/2)||h(x)||² + λ̃ᵀmax(0,g(x)+λ̃/μ) + (μ/2)||max(0,g(x)+λ̃/μ)||²
        """
        n = len(x0)
        x = x0.copy()
        mu = penalty_start
        
        # Inicializar multiplicadores
        if constraints_eq:
            m_eq = len(constraints_eq)
            if lambda_eq0 is None:
                lambda_eq = np.zeros(m_eq)
            else:
                lambda_eq = lambda_eq0.copy()
        else:
            lambda_eq = np.array([])
            m_eq = 0
        
        if constraints_ineq:
            m_ineq = len(constraints_ineq)
            if lambda_ineq0 is None:
                lambda_ineq = np.zeros(m_ineq)
            else:
                lambda_ineq = lambda_ineq0.copy()
        else:
            lambda_ineq = np.array([])
            m_ineq = 0
        
        path_x = [x.copy()]
        path_lambda_eq = [lambda_eq.copy()]
        path_lambda_ineq = [lambda_ineq.copy()]
        errors = [func(*x)]
        violations = []
        
        for outer_iter in range(max_outer_iter):
            # Definir Lagrangiano Aumentado
            def augmented_lagrangian_func(*args):
                x_eval = np.array(args)
                f_val = func(*x_eval)
                aug_term = 0.0
                
                # Términos para restricciones de igualdad
                if constraints_eq:
                    for i, h in enumerate(constraints_eq):
                        h_val = h(x_eval)
                        aug_term += lambda_eq[i] * h_val + (mu/2) * h_val**2
                
                # Términos para restricciones de desigualdad
                if constraints_ineq:
                    for i, g in enumerate(constraints_ineq):
                        g_val = g(x_eval)
                        term = g_val + lambda_ineq[i]/mu
                        if term > 0:
                            aug_term += lambda_ineq[i] * g_val + (mu/2) * g_val**2
                        else:
                            aug_term -= lambda_ineq[i]**2 / (2*mu)
                
                return f_val + aug_term
            
            def augmented_lagrangian_grad(*args):
                x_eval = np.array(args)
                grad_f = grad(*x_eval)
                grad_aug = grad_f.copy()
                
                # Gradiente para restricciones de igualdad
                if constraints_eq:
                    for i, (h, h_jac) in enumerate(zip(constraints_eq, constraints_eq_jac)):
                        h_val = h(x_eval)
                        h_grad = h_jac(x_eval)
                        grad_aug += (lambda_eq[i] + mu * h_val) * h_grad
                
                # Gradiente para restricciones de desigualdad
                if constraints_ineq:
                    for i, (g, g_jac) in enumerate(zip(constraints_ineq, constraints_ineq_jac)):
                        g_val = g(x_eval)
                        g_grad = g_jac(x_eval)
                        term = g_val + lambda_ineq[i]/mu
                        if term > 0:
                            grad_aug += (lambda_ineq[i] + mu * g_val) * g_grad
                
                return grad_aug
            
            # Optimizar Lagrangiano Aumentado
            result = self._unconstrained_optimization(
                augmented_lagrangian_func, augmented_lagrangian_grad, 
                x, max_inner_iter, tol
            )
            x = result['x']
            
            # Actualizar multiplicadores
            if constraints_eq:
                for i, h in enumerate(constraints_eq):
                    h_val = h(x)
                    lambda_eq[i] = lambda_eq[i] + mu * h_val
            
            if constraints_ineq:
                for i, g in enumerate(constraints_ineq):
                    g_val = g(x)
                    lambda_ineq[i] = max(0, lambda_ineq[i] + mu * g_val)
            
            # Calcular violación de restricciones
            violation = 0.0
            if constraints_eq:
                violation += sum([h(x)**2 for h in constraints_eq])
            if constraints_ineq:
                violation += sum([max(0, g(x))**2 for g in constraints_ineq])
            
            violations.append(violation)
            path_x.append(x.copy())
            path_lambda_eq.append(lambda_eq.copy())
            path_lambda_ineq.append(lambda_ineq.copy())
            errors.append(func(*x))
            
            # Verificar convergencia
            if violation < tol:
                break
            
            # Actualizar parámetro de penalización
            mu *= penalty_factor
        
        return {
            'x': x,
            'lambda_eq': lambda_eq,
            'lambda_ineq': lambda_ineq,
            'path': np.array(path_x),  # Usar path estándar
            'path_lambda_eq': np.array(path_lambda_eq),
            'path_lambda_ineq': np.array(path_lambda_ineq),
            'errors': errors,
            'violations': violations,
            'iterations': outer_iter + 1,  # Usar iterations estándar
            'converged': violation < tol,
            'method': 'augmented_lagrangian'
        }
    
    def sequential_quadratic_programming(self, func: Callable, grad: Callable, hess: Callable,
                                       constraints_eq: List[Callable],
                                       constraints_eq_jac: List[Callable],
                                       constraints_ineq: List[Callable] = None,
                                       constraints_ineq_jac: List[Callable] = None,
                                       x0: np.ndarray = None,
                                       max_iter: int = 100,
                                       tol: float = 1e-6) -> Dict[str, Any]:
        """
        Programación Cuadrática Secuencial (SQP)
        
        En cada iteración resuelve:
        min (1/2)dᵀBd + ∇f(x)ᵀd
        s.t. ∇h(x)ᵀd + h(x) = 0
             ∇g(x)ᵀd + g(x) ≤ 0
        """
        n = len(x0)
        x = x0.copy()
        B = np.eye(n)  # Aproximación inicial de la hessiana
        
        m_eq = len(constraints_eq) if constraints_eq else 0
        m_ineq = len(constraints_ineq) if constraints_ineq else 0
        
        path = [x.copy()]
        errors = [func(*x)]
        violations = []
        
        for iteration in range(max_iter):
            f_grad = grad(*x)
            
            # Evaluar restricciones y sus jacobianos
            if constraints_eq:
                h_vals = np.array([h(x) for h in constraints_eq])
                A_eq = np.array([jac(*x) for jac in constraints_eq_jac])
            else:
                h_vals = np.array([])
                A_eq = np.zeros((0, n))
            
            if constraints_ineq:
                g_vals = np.array([g(x) for g in constraints_ineq])
                A_ineq = np.array([jac(*x) for jac in constraints_ineq_jac])
                
                # Identificar restricciones activas
                active_mask = g_vals >= -tol
                g_vals_active = g_vals[active_mask]
                A_ineq_active = A_ineq[active_mask]
            else:
                g_vals_active = np.array([])
                A_ineq_active = np.zeros((0, n))
            
            # Resolver subproblema cuadrático
            direction, multipliers = self._solve_qp_subproblem(
                B, f_grad, A_eq, h_vals, A_ineq_active, g_vals_active
            )
            
            # Verificar condiciones de parada
            violation = 0.0
            if len(h_vals) > 0:
                violation += np.linalg.norm(h_vals)
            if len(g_vals_active) > 0:
                violation += np.linalg.norm(np.maximum(0, g_vals_active))
            
            violations.append(violation)
            
            if np.linalg.norm(direction) < tol and violation < tol:
                break
            
            # Búsqueda de línea
            alpha = self._line_search_sqp(func, constraints_eq, constraints_ineq, x, direction)
            
            # Actualizar posición
            x_new = x + alpha * direction
            
            # Actualizar aproximación de la hessiana (BFGS)
            s = x_new - x
            y = grad(*x_new) - f_grad
            
            # Actualización BFGS
            if np.dot(s, y) > 1e-12:
                rho = 1.0 / np.dot(s, y)
                I = np.eye(n)
                B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            x = x_new
            path.append(x.copy())
            errors.append(func(*x))
        
        return {
            'x': x,
            'path': np.array(path),
            'errors': errors,
            'violations': violations,
            'iterations': iteration + 1,
            'converged': np.linalg.norm(direction) < tol and violation < tol,
            'method': 'sequential_quadratic_programming'
        }
    
    def _line_search_lagrangian(self, func: Callable, constraints: List[Callable],
                               x: np.ndarray, lam: np.ndarray,
                               delta_x: np.ndarray, delta_lambda: np.ndarray) -> float:
        """Búsqueda de línea para el método de Newton del Lagrangiano"""
        alpha = 1.0
        
        for _ in range(20):
            x_new = x + alpha * delta_x
            lam_new = lam + alpha * delta_lambda
            
            # Función de mérito (violación de restricciones)
            violation = sum([h(x_new)**2 for h in constraints])
            
            if violation < sum([h(x)**2 for h in constraints]):
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def _solve_qp_subproblem(self, B: np.ndarray, g: np.ndarray,
                           A_eq: np.ndarray, b_eq: np.ndarray,
                           A_ineq: np.ndarray, b_ineq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve el subproblema cuadrático del SQP
        Implementación simplificada usando método de eliminación
        """
        n = len(g)
        
        if len(A_eq) == 0 and len(A_ineq) == 0:
            # Sin restricciones
            direction = -np.linalg.solve(B, g)
            return direction, np.array([])
        
        # Combinar restricciones activas
        if len(A_eq) > 0 and len(A_ineq) > 0:
            A = np.vstack([A_eq, A_ineq])
            b = np.concatenate([b_eq, b_ineq])
        elif len(A_eq) > 0:
            A = A_eq
            b = b_eq
        else:
            A = A_ineq
            b = b_ineq
        
        # Resolver sistema KKT del subproblema
        m = len(A)
        KKT = np.block([
            [B, A.T],
            [A, np.zeros((m, m))]
        ])
        
        rhs = np.concatenate([-g, -b])
        
        try:
            solution = np.linalg.solve(KKT, rhs)
            direction = solution[:n]
            multipliers = solution[n:]
            return direction, multipliers
        except np.linalg.LinAlgError:
            # Fallback: usar gradiente proyectado
            direction = -g
            if len(A) > 0:
                P = np.eye(n) - A.T @ np.linalg.pinv(A @ A.T) @ A
                direction = P @ direction
            return direction, np.zeros(m)
    
    def _line_search_sqp(self, func: Callable,
                        constraints_eq: List[Callable],
                        constraints_ineq: List[Callable],
                        x: np.ndarray, direction: np.ndarray) -> float:
        """Búsqueda de línea para SQP usando función de mérito"""
        alpha = 1.0
        
        # Función de mérito: f(x) + penalty * ||constraints||
        penalty = 10.0
        
        def merit_function(x_eval):
            f_val = func(*x_eval)
            constraint_violation = 0.0
            
            if constraints_eq:
                constraint_violation += sum([h(x_eval)**2 for h in constraints_eq])
            if constraints_ineq:
                constraint_violation += sum([max(0, g(x_eval))**2 for g in constraints_ineq])
            
            return f_val + penalty * constraint_violation
        
        current_merit = merit_function(x)
        
        for _ in range(20):
            x_new = x + alpha * direction
            new_merit = merit_function(x_new)
            
            if new_merit < current_merit:
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def _unconstrained_optimization(self, func: Callable, grad: Callable, x0: np.ndarray,
                                  max_iter: int, tol: float) -> Dict[str, Any]:
        """Optimización sin restricciones usando gradiente descendente con búsqueda de línea"""
        x = x0.copy()
        path = [x.copy()]
        
        for i in range(max_iter):
            gradient = grad(*x)
            
            if np.linalg.norm(gradient) < tol:
                break
            
            # Búsqueda de línea simple
            alpha = 1.0
            for _ in range(10):
                x_new = x - alpha * gradient
                try:
                    if func(*x_new) < func(*x):
                        break
                except:
                    pass
                alpha *= 0.5
            
            x = x - alpha * gradient
            path.append(x.copy())
        
        return {
            'x': x,
            'path': path,
            'iterations': i + 1
        }
