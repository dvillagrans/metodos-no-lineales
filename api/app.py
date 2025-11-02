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
    """
    API para ejecutar métodos de búsqueda de línea.
    Permite seleccionar entre variantes: sección áurea, búsqueda de Fibonacci y Armijo backtracking.
    Estos métodos buscan el mínimo de una función a lo largo de una dirección dada, optimizando el tamaño de paso.
    """
    try:
        # Extrae los datos enviados en el cuerpo de la petición (JSON)
        data = request.get_json()
        # Selecciona el método de búsqueda de línea a usar (por defecto: golden_section)
        method = data.get('method', 'golden_section')  # Puede ser 'golden_section', 'fibonacci_search' o 'armijo_backtracking'
        # Punto inicial para la búsqueda (vector numpy)
        x0 = np.array(data.get('x0', [0.0, 0.0]))

        # Instancia el optimizador de búsqueda de línea
        optimizer = LineSearchOptimizer()

        # Según el método seleccionado, ejecuta la variante correspondiente:
        if method == 'golden_section':
            # Sección áurea:
            #   - Divide el intervalo usando la proporción áurea para encontrar el mínimo.
            #   - Es eficiente y no requiere derivadas.
            result = optimizer.golden_section(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'fibonacci_search':
            # Búsqueda de Fibonacci:
            #   - Similar a la sección áurea pero usa la secuencia de Fibonacci para subdividir el intervalo.
            #   - Puede ser más precisa en algunos casos.
            result = optimizer.fibonacci_search(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'armijo_backtracking':
            # Armijo backtracking:
            #   - Reduce el tamaño de paso hasta que se cumple la condición de Armijo (suficiente descenso).
            #   - Muy usado en métodos de gradiente para asegurar convergencia.
            result = optimizer.armijo_backtracking(
                func_wrapper, grad_wrapper, x0, None,
                max_iter=100, tol=1e-6
            )
        else:
            # Si el método no es reconocido, regresa un error con código 400 (petición incorrecta)
            return jsonify({'error': 'Método no reconocido'}), 400

        # Devuelve la respuesta estandarizada con el resultado del método seleccionado
        # La respuesta incluye:
        #   - x: el punto óptimo encontrado
        #   - path: la trayectoria de puntos visitados
        #   - errors: la evolución del error en cada iteración
        #   - iterations: número de iteraciones realizadas
        #   - converged: si el método alcanzó la tolerancia
        #   - method: el nombre de la variante utilizada
        return create_response(result)

    except Exception as e:
        # Si ocurre cualquier excepción durante la ejecución del método:
        # 1. Se captura la excepción y se obtiene el traceback para depuración.
        # 2. Se construye un diccionario con información del error, el tipo y el traceback.
        # 3. Se imprime el error en consola para facilitar el debug en el servidor.
        # 4. Se retorna una respuesta JSON con el error y código HTTP 500 (error interno del servidor).
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
    """
    API para ejecutar métodos de descenso de gradiente.
    Permite seleccionar entre variantes: clásico, con momentum y Adam.
    Estos métodos buscan el mínimo de una función usando la dirección del gradiente, con diferentes estrategias para acelerar o estabilizar la convergencia.
    """
    try:
        # Extrae los datos enviados en el cuerpo de la petición (JSON)
        data = request.get_json()
        # Selecciona el método de descenso de gradiente a usar (por defecto: clásico)
        method = data.get('method', 'gradient_descent')  # Puede ser 'gradient_descent', 'gradient_descent_momentum' o 'adam'
        # Punto inicial para la optimización (vector numpy)
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        # Parámetros de configuración
        learning_rate = data.get('learning_rate', 0.1)
        max_iter = data.get('max_iter', 1000)
        tolerance = data.get('tolerance', 1e-6)

        # Instancia el optimizador de descenso de gradiente
        optimizer = GradientDescentOptimizer()

        # Según el método seleccionado, ejecuta la variante correspondiente:
        if method == 'gradient_descent':
            # Descenso de gradiente clásico:
            #   - Da pasos en la dirección opuesta al gradiente.
            #   - Es el método base para optimización sin restricciones.
            result = optimizer.gradient_descent(
                func_wrapper, grad_wrapper, x0,
                learning_rate=learning_rate, max_iter=max_iter, tol=tolerance
            )
        elif method == 'gradient_descent_momentum':
            # Descenso de gradiente con momentum:
            #   - Acumula una fracción del paso anterior para acelerar la convergencia y evitar mínimos locales poco profundos.
            #   - El parámetro momentum (por defecto 0.9) controla la inercia.
            result = optimizer.gradient_descent_momentum(
                func_wrapper, grad_wrapper, x0,
                learning_rate=learning_rate, momentum=0.9, max_iter=max_iter, tol=tolerance
            )
        elif method == 'adam':
            # Adam (Adaptive Moment Estimation):
            #   - Algoritmo adaptativo que ajusta el learning rate para cada parámetro usando promedios móviles de gradientes y sus cuadrados.
            #   - Muy popular en machine learning por su robustez y velocidad.
            result = optimizer.adam(
                func_wrapper, grad_wrapper, x0,
                learning_rate=learning_rate, max_iter=max_iter, tol=tolerance
            )
        else:
            # Si el método no es reconocido, regresa un error con código 400 (petición incorrecta)
            return jsonify({'error': 'Método no reconocido'}), 400

        # Devuelve la respuesta estandarizada con el resultado del método seleccionado
        # La respuesta incluye:
        #   - x: el punto óptimo encontrado
        #   - path: la trayectoria de puntos visitados
        #   - errors: la evolución del error en cada iteración
        #   - iterations: número de iteraciones realizadas
        #   - converged: si el método alcanzó la tolerancia
        #   - method: el nombre de la variante utilizada
        return create_response(result)

    except Exception as e:
        # Si ocurre cualquier excepción durante la ejecución del método:
        # 1. Se captura la excepción y se obtiene el traceback para depuración.
        # 2. Se construye un diccionario con información del error, el tipo y el traceback.
        # 3. Se imprime el error en consola para facilitar el debug en el servidor.
        # 4. Se retorna una respuesta JSON con el error y código HTTP 500 (error interno del servidor).
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'type': type(e).__name__
        }
        print(f"Error en run_gradient_descent: {error_info}")
        return jsonify(error_info), 500


# -----------------------------
# API para ejecutar el Método de Newton
# -----------------------------
@app.route('/api/run_newton', methods=['POST'])
def api_run_newton():
    """
    API para ejecutar el Método de Newton (y variantes) para optimización no lineal.
    Utiliza condiciones de segundo orden (gradiente y hessiana).
    """
    try:
        data = request.get_json()
        # Selecciona el método de Newton a usar
        method = data.get('method', 'newton_method')
        # Punto inicial para la optimización
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        # Número máximo de iteraciones
        max_iter = data.get('max_iter', 100)
        # Tolerancia para criterio de parada
        tolerance = data.get('tolerance', 1e-6)

        # Instancia el optimizador de Newton
        optimizer = NewtonOptimizer()

        # Ejecuta la variante seleccionada del método de Newton
        if method == 'newton_method':
            # Método de Newton clásico: usa gradiente y hessiana
            result = optimizer.newton_method(
                func_wrapper, grad_wrapper, hess_wrapper, x0,
                max_iter=max_iter, tol=tolerance
            )
        elif method == 'modified_newton':
            # Método de Newton modificado (puede modificar la hessiana para asegurar descenso)
            result = optimizer.modified_newton(
                func_wrapper, grad_wrapper, hess_wrapper, x0,
                max_iter=max_iter, tol=tolerance
            )
        elif method == 'damped_newton':
            # Método de Newton amortiguado (hace pasos más pequeños para mayor robustez)
            result = optimizer.damped_newton(
                func_wrapper, grad_wrapper, hess_wrapper, x0,
                max_iter=max_iter, tol=tolerance
            )
        else:
            # Si el método no es reconocido, regresa error
            return jsonify({'error': 'Método no reconocido'}), 400

        # Devuelve la respuesta estandarizada con el resultado
        return create_response(result)

    except Exception as e:
        # Manejo de errores y logging
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
    """
    API para ejecutar el método de Gradiente Conjugado.
    Permite seleccionar entre variantes clásicas: Fletcher-Reeves, Polak-Ribiere y Hestenes-Stiefel.
    Este método es útil para problemas de optimización sin restricciones y aprovecha la información de gradientes previos para acelerar la convergencia.
    """
    try:
        # Extrae los datos enviados en el cuerpo de la petición (JSON)
        data = request.get_json()

        # Selecciona el método de gradiente conjugado a usar (por defecto: Fletcher-Reeves)
        method = data.get('method', 'fletcher_reeves')  # Puede ser 'fletcher_reeves', 'polak_ribiere' o 'hestenes_stiefel'

        # Punto inicial para la optimización (vector numpy)
        x0 = np.array(data.get('x0', [0.0, 0.0]))

        # Instancia el optimizador de gradiente conjugado, que contiene las variantes implementadas
        optimizer = ConjugateGradientOptimizer()

        # Según el método seleccionado, ejecuta la variante correspondiente:
        if method == 'fletcher_reeves':
            # Variante Fletcher-Reeves:
            #   - Calcula el parámetro beta como el cociente de las normas cuadradas de los gradientes sucesivos.
            #   - Es la versión más clásica y robusta para funciones cuadráticas.
            result = optimizer.fletcher_reeves(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'polak_ribiere':
            # Variante Polak-Ribiere:
            #   - Beta se calcula usando el producto escalar entre el gradiente actual y la diferencia de gradientes.
            #   - Puede ser más eficiente en algunos problemas no cuadráticos.
            result = optimizer.polak_ribiere(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'hestenes_stiefel':
            # Variante Hestenes-Stiefel:
            #   - Beta se calcula usando la diferencia entre gradientes y la dirección de búsqueda anterior.
            #   - Suele ser útil cuando la función objetivo no es estrictamente cuadrática.
            result = optimizer.hestenes_stiefel(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        else:
            # Si el método no es reconocido, regresa un error con código 400 (petición incorrecta)
            return jsonify({'error': 'Método no reconocido'}), 400

        # Devuelve la respuesta estandarizada con el resultado del método seleccionado
        # La respuesta incluye:
        #   - x: el punto óptimo encontrado
        #   - path: la trayectoria de puntos visitados
        #   - errors: la evolución del error en cada iteración
        #   - iterations: número de iteraciones realizadas
        #   - converged: si el método alcanzó la tolerancia
        #   - method: el nombre de la variante utilizada
        return create_response(result)

    except Exception as e:
        # Si ocurre cualquier excepción durante la ejecución del método:
        # 1. Se captura la excepción y se obtiene el traceback para depuración.
        # 2. Se construye un diccionario con información del error, el tipo y el traceback.
        # 3. Se imprime el error en consola para facilitar el debug en el servidor.
        # 4. Se retorna una respuesta JSON con el error y código HTTP 500 (error interno del servidor).
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
    """
    API para ejecutar métodos de Cuasi-Newton.
    Permite seleccionar entre variantes populares: BFGS, DFP y L-BFGS.
    Los métodos cuasi-Newton aproximan la matriz Hessiana (o su inversa) usando solo gradientes, logrando eficiencia y robustez en problemas de optimización sin restricciones.
    """
    try:
        # Extrae los datos enviados en el cuerpo de la petición (JSON)
        data = request.get_json()
        # Selecciona el método cuasi-Newton a usar (por defecto: BFGS)
        method = data.get('method', 'bfgs')  # Puede ser 'bfgs', 'dfp' o 'l_bfgs'
        # Punto inicial para la optimización (vector numpy)
        x0 = np.array(data.get('x0', [0.0, 0.0]))

        # Instancia el optimizador de cuasi-Newton, que contiene las variantes implementadas
        optimizer = QuasiNewtonOptimizer()

        # Según el método seleccionado, ejecuta la variante correspondiente:
        if method == 'bfgs':
            # BFGS (Broyden-Fletcher-Goldfarb-Shanno):
            #   - Actualiza una aproximación de la inversa de la Hessiana usando información de gradientes y desplazamientos.
            #   - Es el método cuasi-Newton más usado por su estabilidad y eficiencia.
            result = optimizer.bfgs(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'dfp':
            # DFP (Davidon-Fletcher-Powell):
            #   - Primer método cuasi-Newton ampliamente utilizado.
            #   - También actualiza la inversa de la Hessiana, pero con una fórmula diferente a BFGS.
            result = optimizer.dfp(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        elif method == 'l_bfgs':
            # L-BFGS (Limited-memory BFGS):
            #   - Variante de BFGS que almacena solo un número limitado de vectores para problemas de gran dimensión.
            #   - Muy eficiente en machine learning y optimización a gran escala.
            result = optimizer.l_bfgs(
                func_wrapper, grad_wrapper, x0,
                max_iter=100, tol=1e-6
            )
        else:
            # Si el método no es reconocido, regresa un error con código 400 (petición incorrecta)
            return jsonify({'error': 'Método no reconocido'}), 400

        # Devuelve la respuesta estandarizada con el resultado del método seleccionado
        # La respuesta incluye:
        #   - x: el punto óptimo encontrado
        #   - path: la trayectoria de puntos visitados
        #   - errors: la evolución del error en cada iteración
        #   - iterations: número de iteraciones realizadas
        #   - converged: si el método alcanzó la tolerancia
        #   - method: el nombre de la variante utilizada
        return create_response(result)

    except Exception as e:
        # Si ocurre cualquier excepción durante la ejecución del método:
        # 1. Se captura la excepción y se obtiene el traceback para depuración.
        # 2. Se construye un diccionario con información del error, el tipo y el traceback.
        # 3. Se imprime el error en consola para facilitar el debug en el servidor.
        # 4. Se retorna una respuesta JSON con el error y código HTTP 500 (error interno del servidor).
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
        # Robust parsing for max_iter and tolerance
        max_iter_raw = data.get('max_iter', None)
        tolerance_raw = data.get('tolerance', 1e-6)
        try:
            if max_iter_raw is None or str(max_iter_raw).strip() == '':
                max_iter = 1_000_000  # Sin límite práctico
            else:
                max_iter = int(max_iter_raw)
        except Exception:
            max_iter = 1_000_000
        try:
            tolerance = float(tolerance_raw) if tolerance_raw is not None else 1e-6
        except Exception:
            tolerance = 1e-6
        constraint_type = data.get('constraint_type', 'linear_ineq')
        
        # Configurar restricciones según el tipo seleccionado
        if constraint_type == 'linear_ineq':
            # Restricción lineal: x + y <= 3
            constraints = {
                'type': 'ineq',
                'fun': lambda x: 3 - x[0] - x[1],
                'jac': lambda x: np.array([-1, -1])
            }
            # Proyección si el punto inicial no es factible
            if constraints['fun'](x0) < 0:
                # Proyectar sobre x + y = 3 (borde)
                s = (x0[0] + x0[1] - 3) / 2
                x0 = np.array([x0[0] - s, x0[1] - s])
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
            # Proyección si el punto inicial no es factible
            if constraints['fun'](x0) < 0:
                # Proyectar radialmente sobre el círculo x^2 + y^2 = 4
                norm = np.linalg.norm(x0)
                if norm == 0:
                    x0 = np.array([2.0, 0.0])
                else:
                    x0 = x0 * (2.0 / norm)
        else:
            # Sin restricciones (caso por defecto)
            constraints = None
        

        optimizer = ConstrainedOptimizer()
        learning_rate = data.get('learning_rate', 0.01)

        # Si es restricción de igualdad y método gradiente proyectado, proyectar el punto después de cada iteración
        if method == 'projected_gradient' and constraint_type == 'linear_eq':
            def project_onto_linear_eq(x):
                # Proyecta x sobre x[0] + x[1] = 2
                s = (x[0] + x[1] - 2) / 2
                return np.array([x[0] - s, x[1] - s])

            def projected_gradient_with_projection(func, grad, x0, constraints, learning_rate=0.01, max_iter=1000, tol=1e-6):
                x = np.copy(x0)
                path = [x.copy()]
                errors = []
                converged = False
                for i in range(max_iter):
                    grad_val = grad(x)
                    x_new = x - learning_rate * grad_val
                    # Proyectar sobre la restricción de igualdad
                    x_new = project_onto_linear_eq(x_new)
                    path.append(x_new.copy())
                    err = np.linalg.norm(x_new - x)
                    errors.append(err)
                    if err < tol:
                        converged = True
                        break
                    x = x_new
                result = {
                    'x': x,
                    'path': np.array(path),
                    'errors': np.array(errors),
                    'iterations': i+1,
                    'converged': converged,
                    'method': 'projected_gradient_with_projection'
                }
                return result

            result = projected_gradient_with_projection(
                func_wrapper, grad_wrapper, x0,
                constraints, learning_rate=learning_rate, max_iter=max_iter, tol=tolerance
            )
        # Si es restricción no lineal (círculo) y método gradiente proyectado, proyectar el punto sobre la frontera después de cada iteración
        elif method == 'projected_gradient' and constraint_type == 'nonlinear_ineq':
            # --- MÉTODO: Gradiente proyectado para restricción no lineal (círculo x^2 + y^2 <= 4) ---

            # Función que proyecta un punto sobre el círculo de radio 2 si se sale fuera
            def project_onto_circle(x):
                norm = np.linalg.norm(x)
                if norm == 0:
                    # Si el punto es el origen, lo mandamos a (2,0) (en la frontera)
                    return np.array([2.0, 0.0])
                if norm > 2.0:
                    # Si el punto está fuera del círculo, lo escalamos a la frontera
                    return x * (2.0 / norm)
                # Si está dentro, lo dejamos igual
                return x

            # Algoritmo de gradiente proyectado con proyección sobre el círculo en cada iteración
            def projected_gradient_with_circle(func, grad, x0, constraints, learning_rate=0.01, max_iter=1000, tol=1e-6):
                x = np.copy(x0)  # Copia del punto inicial
                path = [x.copy()]  # Lista para guardar la trayectoria de puntos
                errors = []        # Lista para guardar el error en cada paso
                converged = False  # Bandera de convergencia
                for i in range(max_iter):
                    grad_val = grad(x)  # Calcula el gradiente en el punto actual
                    x_new = x - learning_rate * grad_val  # Da un paso de descenso
                    # Proyecta el nuevo punto sobre el círculo si sale fuera
                    x_new = project_onto_circle(x_new)
                    path.append(x_new.copy())  # Guarda el nuevo punto en la trayectoria
                    err = np.linalg.norm(x_new - x)  # Calcula el error (distancia entre puntos)
                    errors.append(err)
                    if err < tol:
                        # Si el error es menor que la tolerancia, se considera convergido
                        converged = True
                        break
                    x = x_new  # Actualiza el punto para la siguiente iteración
                # Prepara el resultado en formato dict
                result = {
                    'x': x,
                    'path': np.array(path),
                    'errors': np.array(errors),
                    'iterations': i+1,
                    'converged': converged,
                    'method': 'projected_gradient_with_circle'
                }
                return result

            # Ejecuta el método y obtiene el resultado final
            result = projected_gradient_with_circle(
                func_wrapper, grad_wrapper, x0,
                constraints, learning_rate=learning_rate, max_iter=max_iter, tol=tolerance
            )
        elif method == 'projected_gradient':
            result = optimizer.projected_gradient(
                func_wrapper, grad_wrapper, x0,
                constraints, learning_rate=learning_rate, max_iter=max_iter, tol=tolerance
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
    """
    API para ejecutar métodos de penalización y barrera.
    Permite seleccionar entre variantes: penalización exterior, barrera logarítmica y lagrangiano aumentado.
    Estos métodos transforman un problema restringido en uno sin restricciones, agregando penalizaciones o barreras a la función objetivo para forzar el cumplimiento de las restricciones.
    """
    try:
        # Extrae los datos enviados en el cuerpo de la petición (JSON)
        data = request.get_json()
        # Selecciona el método de penalización/barrera a usar (por defecto: exterior_penalty)
        method = data.get('method', 'exterior_penalty')  # Puede ser 'exterior_penalty', 'logarithmic_barrier' o 'augmented_lagrangian'
        # Punto inicial para la optimización (vector numpy)
        x0 = np.array(data.get('x0', [0.0, 0.0]))
        # Parámetros de configuración
        max_iter = data.get('max_iter', 100)
        tolerance = data.get('tolerance', 1e-6)
        constraint_type = data.get('constraint_type', 'linear_ineq')
        penalty_param = data.get('penalty_param', 1.0)

        # Configurar restricciones según el tipo seleccionado
        constraints_ineq = []
        if constraint_type == 'linear_ineq':
            # Restricción lineal: x + y <= 3 -> 3 - x - y >= 0
            constraints_ineq = [lambda x: 3 - x[0] - x[1]]
        elif constraint_type == 'nonlinear_ineq':
            # Restricción no lineal: x² + y² <= 4 -> 4 - x² - y² >= 0
            constraints_ineq = [lambda x: 4 - x[0]**2 - x[1]**2]
        elif constraint_type == 'box_constraints':
            # Restricciones de caja: 0 <= x <= 2, 0 <= y <= 2
            constraints_ineq = [
                lambda x: x[0],      # x >= 0
                lambda x: 2 - x[0],  # x <= 2
                lambda x: x[1],      # y >= 0
                lambda x: 2 - x[1]   # y <= 2
            ]

        # Instancia el optimizador de penalización y barrera
        optimizer = PenaltyBarrierOptimizer()

        # Según el método seleccionado, ejecuta la variante correspondiente:
        if method == 'exterior_penalty':
            # Penalización exterior:
            #   - Suma un término cuadrático a la función objetivo que penaliza la violación de las restricciones.
            #   - El parámetro penalty_start controla la severidad de la penalización.
            result = optimizer.exterior_penalty(
                func_wrapper, grad_wrapper,
                constraints_ineq=constraints_ineq,
                x0=x0, penalty_start=penalty_param, 
                max_outer_iter=20, max_inner_iter=max_iter, tol=tolerance
            )
        elif method == 'logarithmic_barrier':
            # Barrera logarítmica:
            #   - Suma una barrera logarítmica a la función objetivo que tiende a infinito al acercarse a la frontera factible.
            #   - El parámetro barrier_start controla la fuerza de la barrera.
            result = optimizer.logarithmic_barrier(
                func_wrapper, grad_wrapper, constraints_ineq,
                x0, barrier_start=penalty_param, 
                max_outer_iter=20, max_inner_iter=max_iter, tol=tolerance
            )
        elif method == 'augmented_lagrangian':
            # Lagrangiano aumentado (o penalización mixta):
            #   - Combina penalización cuadrática y multiplicadores de Lagrange para mejorar la convergencia.
            #   - Si no existe augmented_lagrangian, se usa mixed_penalty_barrier.
            result = optimizer.mixed_penalty_barrier(
                func_wrapper, grad_wrapper,
                constraints_ineq=constraints_ineq,
                x0=x0, penalty_start=penalty_param,
                max_outer_iter=20, max_inner_iter=max_iter, tol=tolerance
            )
        else:
            # Si el método no es reconocido, regresa un error con código 400 (petición incorrecta)
            return jsonify({'error': 'Método no reconocido'}), 400

        # Devuelve la respuesta estandarizada con el resultado del método seleccionado
        # La respuesta incluye:
        #   - x: el punto óptimo encontrado
        #   - path: la trayectoria de puntos visitados
        #   - errors: la evolución del error en cada iteración
        #   - iterations: número de iteraciones realizadas
        #   - converged: si el método alcanzó la tolerancia
        #   - method: el nombre de la variante utilizada
        return create_response(result)

    except Exception as e:
        # Si ocurre cualquier excepción durante la ejecución del método:
        # 1. Se captura la excepción y se obtiene el traceback para depuración.
        # 2. Se construye un diccionario con información del error, el tipo y el traceback.
        # 3. Se imprime el error en consola para facilitar el debug en el servidor.
        # 4. Se retorna una respuesta JSON con el error y código HTTP 500 (error interno del servidor).
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
