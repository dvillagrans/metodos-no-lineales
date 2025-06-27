"""
Aplicación Flask para Métodos de Optimización No Lineal
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from metodos import (
    GradientDescentOptimizer,
    NewtonOptimizer,
    QuasiNewtonOptimizer,
    LineSearchOptimizer,
    ConjugateGradientOptimizer,
    ConstrainedOptimizer,
    PenaltyBarrierOptimizer,
    LagrangeMultiplierOptimizer
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'optimization_methods_2025'

# Funciones de ejemplo para las demostraciones
def ejemplo_funcion_objetivo(x, y):
    """Función de prueba: f(x,y) = (x-1)² + (y-2)²"""
    return (x - 1)**2 + (y - 2)**2

def ejemplo_gradiente(x, y):
    """Gradiente de la función de prueba"""
    return np.array([2*(x - 1), 2*(y - 2)])

def ejemplo_hessiana(x, y):
    """Hessiana de la función de prueba"""
    return np.array([[2, 0], [0, 2]])

# Wrappers para funciones que reciben arrays
def func_wrapper(*args):
    """Wrapper para función objetivo que puede recibir array o argumentos separados"""
    if len(args) == 1 and hasattr(args[0], '__len__') and len(args[0]) == 2:
        # Si recibe un array numpy [x, y]
        x_array = args[0]
        return ejemplo_funcion_objetivo(x_array[0], x_array[1])
    elif len(args) == 2:
        # Si recibe argumentos separados func(x, y)
        return ejemplo_funcion_objetivo(args[0], args[1])
    else:
        raise ValueError(f"func_wrapper expects 1 array or 2 arguments, got {len(args)}")

def grad_wrapper(*args):
    """Wrapper para gradiente que puede recibir array o argumentos separados"""
    if len(args) == 1 and hasattr(args[0], '__len__') and len(args[0]) == 2:
        # Si recibe un array numpy [x, y]
        x_array = args[0]
        return ejemplo_gradiente(x_array[0], x_array[1])
    elif len(args) == 2:
        # Si recibe argumentos separados grad(x, y)
        return ejemplo_gradiente(args[0], args[1])
    else:
        raise ValueError(f"grad_wrapper expects 1 array or 2 arguments, got {len(args)}")

def hess_wrapper(*args):
    """Wrapper para hessiana que puede recibir array o argumentos separados"""
    if len(args) == 1 and hasattr(args[0], '__len__') and len(args[0]) == 2:
        # Si recibe un array numpy [x, y]
        x_array = args[0]
        return ejemplo_hessiana(x_array[0], x_array[1])
    elif len(args) == 2:
        # Si recibe argumentos separados hess(x, y)
        return ejemplo_hessiana(args[0], args[1])
    else:
        raise ValueError(f"hess_wrapper expects 1 array or 2 arguments, got {len(args)}")

def restriccion_igualdad(x, y):
    """Restricción de igualdad: x + y - 2 = 0"""
    return x + y - 2

def restriccion_igualdad_jac(x, y):
    """Jacobiano de la restricción de igualdad"""
    return np.array([1, 1])

def restriccion_desigualdad(x, y):
    """Restricción de desigualdad: x² + y² - 4 ≤ 0"""
    return x**2 + y**2 - 4

def restriccion_desigualdad_jac(x, y):
    """Jacobiano de la restricción de desigualdad"""
    return np.array([2*x, 2*y])

def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos Python nativos para serialización JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def create_response(result):
    """Crea una respuesta JSON estandarizada para todos los métodos"""
    return jsonify({
        'x': convert_numpy_types(result['x']),
        'path': convert_numpy_types(result['path']),
        'errors': convert_numpy_types(result['errors']),
        'iterations': convert_numpy_types(result['iterations']),
        'converged': convert_numpy_types(result['converged']),
        'method': result['method']
    })

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/line_search')
def line_search():
    """Página de métodos de búsqueda de línea"""
    return render_template('line_search.html')

@app.route('/gradient_descent')
def gradient_descent():
    """Página de descenso de gradiente"""
    return render_template('gradient_descent.html')

@app.route('/newton_method')
def newton_method():
    """Página del método de Newton"""
    return render_template('newton_method.html')

@app.route('/constrained')
def constrained():
    """Página de métodos primales (gradiente restringido)"""
    return render_template('constrained.html')

@app.route('/penalty_barrier')
def penalty_barrier():
    """Página de métodos de penalización y barrera"""
    return render_template('penalty_barrier.html')

@app.route('/lagrange')
def lagrange():
    """Página de multiplicadores de Lagrange"""
    return render_template('lagrange.html')

# APIs para ejecutar los métodos
@app.route('/api/run_line_search', methods=['POST'])
def api_run_line_search():
    """API para ejecutar búsqueda de línea"""
    try:
        data = request.get_json()
        method = data.get('method', 'golden_section')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        
        optimizer = LineSearchOptimizer()
        
        if method == 'golden_section':
            result = optimizer.golden_section(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'fibonacci_search':
            result = optimizer.fibonacci_search(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'armijo_backtracking':
            result = optimizer.armijo_backtracking(
                func_wrapper, grad_wrapper, x0, None,
                max_iter=100, tol=1e-6
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        # Agregar más información de debug
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_line_search: {error_info}")  # Para logs del servidor
        return jsonify(error_info), 500

@app.route('/api/run_gradient_descent', methods=['POST'])
def api_run_gradient_descent():
    """API para ejecutar descenso de gradiente"""
    try:
        data = request.get_json()
        method = data.get('method', 'gradient_descent')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        learning_rate = data.get('learning_rate', 0.1)
        
        optimizer = GradientDescentOptimizer()
        
        if method == 'gradient_descent':
            result = optimizer.gradient_descent(
                func_wrapper, grad_wrapper, x0,
                learning_rate=learning_rate, max_iter=100, tol=1e-6
            )
        elif method == 'gradient_descent_momentum':
            result = optimizer.gradient_descent_momentum(
                func_wrapper, grad_wrapper, x0,
                learning_rate=learning_rate, momentum=0.9, max_iter=100, tol=1e-6
            )
        elif method == 'adam':
            result = optimizer.adam(
                func_wrapper, grad_wrapper, x0,
                learning_rate=learning_rate, max_iter=100, tol=1e-6
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_gradient_descent: {error_info}")
        return jsonify(error_info), 500

@app.route('/api/run_newton', methods=['POST'])
def api_run_newton():
    """API para ejecutar método de Newton"""
    try:
        data = request.get_json()
        method = data.get('method', 'newton_method')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        max_iter = data.get('max_iter', 100)
        tolerance = data.get('tolerance', 1e-6)
        
        optimizer = NewtonOptimizer()
        
        if method == 'newton_method':
            result = optimizer.newton_method(
                func_wrapper, grad_wrapper, hess_wrapper, x0,
                max_iter=max_iter, tol=tolerance
            )
        elif method == 'modified_newton':
            result = optimizer.modified_newton(
                func_wrapper, grad_wrapper, hess_wrapper, x0,
                max_iter=max_iter, tol=tolerance
            )
        elif method == 'damped_newton':
            result = optimizer.damped_newton(
                func_wrapper, grad_wrapper, hess_wrapper, x0,
                max_iter=max_iter, tol=tolerance
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_newton: {error_info}")
        return jsonify(error_info), 500

@app.route('/api/run_conjugate_gradient', methods=['POST'])
def api_run_conjugate_gradient():
    """API para ejecutar gradiente conjugado"""
    try:
        data = request.get_json()
        method = data.get('method', 'fletcher_reeves')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        
        optimizer = ConjugateGradientOptimizer()
        
        if method == 'fletcher_reeves':
            result = optimizer.fletcher_reeves(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'polak_ribiere':
            result = optimizer.polak_ribiere(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'hestenes_stiefel':
            result = optimizer.hestenes_stiefel(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_conjugate_gradient: {error_info}")
        return jsonify(error_info), 500

@app.route('/api/run_quasi_newton', methods=['POST'])
def api_run_quasi_newton():
    """API para ejecutar cuasi-Newton"""
    try:
        data = request.get_json()
        method = data.get('method', 'bfgs')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        
        optimizer = QuasiNewtonOptimizer()
        
        if method == 'bfgs':
            result = optimizer.bfgs(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'dfp':
            result = optimizer.dfp(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'l_bfgs':
            result = optimizer.l_bfgs(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_quasi_newton: {error_info}")
        return jsonify(error_info), 500

@app.route('/api/run_constrained', methods=['POST'])
def api_run_constrained():
    """API para ejecutar optimización restringida"""
    try:
        data = request.get_json()
        method = data.get('method', 'projected_gradient')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        max_iter = data.get('max_iter', 100)
        tolerance = data.get('tolerance', 1e-6)
        constraint_type = data.get('constraint_type', 'linear_ineq')
        
        # Configurar restricciones según el tipo seleccionado
        if constraint_type == 'linear_ineq':
            # Restricción lineal: x + y <= 3
            constraints = {
                'type': 'ineq',
                'fun': lambda x: 3 - x[0] - x[1],
                'jac': lambda x: np.array([-1, -1])
            }
        elif constraint_type == 'linear_eq':
            # Restricción de igualdad: x + y = 2
            constraints = {
                'type': 'eq',
                'fun': lambda x: x[0] + x[1] - 2,
                'jac': lambda x: np.array([1, 1])
            }
        elif constraint_type == 'nonlinear_ineq':
            # Restricción no lineal: x² + y² <= 4
            constraints = {
                'type': 'ineq',
                'fun': lambda x: 4 - x[0]**2 - x[1]**2,
                'jac': lambda x: np.array([-2*x[0], -2*x[1]])
            }
        else:
            # Sin restricciones (caso por defecto)
            constraints = None
        
        optimizer = ConstrainedOptimizer()
        
        if method == 'projected_gradient':
            result = optimizer.projected_gradient(
                func_wrapper, grad_wrapper, x0,
                constraints, max_iter=max_iter, tol=tolerance
            )
        elif method == 'reduced_gradient':
            result = optimizer.reduced_gradient(
                func_wrapper, grad_wrapper, x0,
                constraints, max_iter=max_iter, tol=tolerance
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_constrained: {error_info}")
        return jsonify(error_info), 500

@app.route('/api/run_penalty_barrier', methods=['POST'])
def api_run_penalty_barrier():
    """API para ejecutar métodos de penalización y barrera"""
    try:
        data = request.get_json()
        method = data.get('method', 'exterior_penalty')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        max_iter = data.get('max_iter', 100)
        tolerance = data.get('tolerance', 1e-6)
        constraint_type = data.get('constraint_type', 'linear_ineq')
        penalty_param = data.get('penalty_param', 1.0)
        
        # Configurar restricciones según el tipo seleccionado
        if constraint_type == 'linear_ineq':
            # Restricción lineal: x + y <= 3
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 3 - x[0] - x[1]}
            ]
        elif constraint_type == 'nonlinear_ineq':
            # Restricción no lineal: x² + y² <= 4
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 4 - x[0]**2 - x[1]**2}
            ]
        elif constraint_type == 'box_constraints':
            # Restricciones de caja: 0 <= x <= 2, 0 <= y <= 2
            constraints = [
                {'type': 'ineq', 'fun': lambda x: x[0]},      # x >= 0
                {'type': 'ineq', 'fun': lambda x: 2 - x[0]},  # x <= 2
                {'type': 'ineq', 'fun': lambda x: x[1]},      # y >= 0
                {'type': 'ineq', 'fun': lambda x: 2 - x[1]}   # y <= 2
            ]
        else:
            constraints = []
        
        optimizer = PenaltyBarrierOptimizer()
        
        if method == 'exterior_penalty':
            result = optimizer.exterior_penalty(
                func_wrapper, x0, constraints,
                penalty_param=penalty_param, max_iter=max_iter, tol=tolerance
            )
        elif method == 'logarithmic_barrier':
            result = optimizer.logarithmic_barrier(
                func_wrapper, x0, constraints,
                barrier_param=penalty_param, max_iter=max_iter, tol=tolerance
            )
        elif method == 'augmented_lagrangian':
            result = optimizer.augmented_lagrangian(
                func_wrapper, x0, constraints,
                penalty_param=penalty_param, max_iter=max_iter, tol=tolerance
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_penalty_barrier: {error_info}")
        return jsonify(error_info), 500

@app.route('/api/run_lagrange', methods=['POST'])
def api_run_lagrange():
    """API para ejecutar multiplicadores de Lagrange"""
    try:
        data = request.get_json()
        method = data.get('method', 'lagrange_newton')
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        max_iter = data.get('max_iter', 100)
        tolerance = data.get('tolerance', 1e-6)
        constraint_type = data.get('constraint_type', 'linear_eq')
        
        # Configurar restricciones según el tipo seleccionado
        if constraint_type == 'linear_eq':
            # Restricción de igualdad: x + y = 2
            constraints_eq = [lambda x: x[0] + x[1] - 2]
            constraints_eq_jac = [lambda x: np.array([1, 1])]
        elif constraint_type == 'nonlinear_eq':
            # Restricción no lineal: x² + y² = 1
            constraints_eq = [lambda x: x[0]**2 + x[1]**2 - 1]
            constraints_eq_jac = [lambda x: np.array([2*x[0], 2*x[1]])]
        elif constraint_type == 'mixed_constraints':
            # Restricción mixta: x + y = 2 y x >= 0, y >= 0
            constraints_eq = [lambda x: x[0] + x[1] - 2]
            constraints_eq_jac = [lambda x: np.array([1, 1])]
        else:
            constraints_eq = []
            constraints_eq_jac = []
        
        optimizer = LagrangeMultiplierOptimizer()
        
        if method == 'lagrange_newton':
            result = optimizer.lagrange_newton(
                func_wrapper, grad_wrapper, hess_wrapper,
                constraints_eq, constraints_eq_jac, x0=x0,
                max_iter=max_iter, tol=tolerance
            )
        elif method == 'augmented_lagrangian':
            result = optimizer.augmented_lagrangian(
                func_wrapper, grad_wrapper, x0,
                constraints_eq, max_iter=max_iter, tol=tolerance
            )
        elif method == 'penalty_lagrange':
            result = optimizer.penalty_lagrange(
                func_wrapper, grad_wrapper, x0,
                constraints_eq, max_iter=max_iter, tol=tolerance
            )
        else:
            return jsonify({'error': 'Método no reconocido'}), 400
        
        return create_response(result)
        
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_lagrange: {error_info}")
        return jsonify(error_info), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
