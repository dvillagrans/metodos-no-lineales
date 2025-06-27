# 🚀 Aplicación Web - Métodos de Optimización No Lineal

Una aplicación Flask interactiva que implementa y visualiza los principales algoritmos de optimización no lineal.

## 📋 Métodos Implementados

### ✅ **Completamente Funcionales:**

1. **Búsqueda de Línea** - Sección áurea, Fibonacci, Armijo
2. **Descenso de Gradiente** - Básico, Momentum, Adam
3. **Método de Newton** - Clásico, Modificado, Amortiguado
4. **Métodos Primales** - Gradiente proyectado, Reducido
5. **Penalización y Barrera** - Exterior, Logarítmica, Inversa
6. **Multiplicadores de Lagrange** - KKT, Lagrangiano aumentado

### 🎯 **Funciones de Prueba:**

- Función cuadrática: `f(x,y) = (x-1)² + (y-2)²`
- Restricciones de igualdad: `x + y - 2 = 0`
- Restricciones de desigualdad: `x² + y² - 4 ≤ 0`

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o navegar al directorio del proyecto:**

   ```bash
   cd c:\Users\diego\workspace\matematicas-avanzadas\No-Lineal\project
   ```

2. **Crear un entorno virtual (recomendado):**

   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual:**
   - **Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

4. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Ejecución de la Aplicación

### Desarrollo

```bash
python app.py
```

### Producción

```bash
flask run --host=0.0.0.0 --port=5000
```

La aplicación estará disponible en: `http://localhost:5000`

## 📁 Estructura del Proyecto

```
project/
├── app.py                  # Aplicación Flask principal
├── requirements.txt        # Dependencias de Python
├── README.md              # Este archivo
├── ejemplo_completo.py    # Demostración de todos los métodos
├── metodos/               # Módulo de algoritmos de optimización
│   ├── __init__.py
│   ├── line_search.py
│   ├── gradient_descent.py
│   ├── newton_method.py
│   ├── conjugate_gradient.py
│   ├── quasi_newton.py
│   ├── constrained_optimization.py
│   ├── penalty_barrier.py
│   └── lagrange_multipliers.py
├── templates/             # Plantillas HTML
│   ├── base.html         # Plantilla base
│   ├── index.html        # Página principal
│   ├── line_search.html  # Búsqueda de línea (FUNCIONAL)
│   ├── gradient_descent.html
│   ├── newton_method.html
│   ├── constrained.html
│   ├── penalty_barrier.html
│   └── lagrange.html
└── static/               # Archivos estáticos
    ├── css/
    │   └── style.css     # Estilos personalizados
    └── js/               # JavaScript (futuro)
```

## 🌟 Características

### 🎨 **Interfaz de Usuario**

- Diseño moderno y responsivo con Bootstrap 5
- Navegación intuitiva con dropdown organizado
- Iconos Font Awesome para mejor UX
- Fórmulas matemáticas renderizadas con MathJax

### 📊 **Visualización**

- Gráficos interactivos con Chart.js
- Convergencia en tiempo real
- Trayectoria de optimización
- Resultados detallados en tablas

### ⚙️ **Funcionalidad**

- Parámetros ajustables (punto inicial, tolerancia, etc.)
- API REST para métodos de optimización
- Manejo de errores robusto
- Indicadores de estado visual

## 🔧 APIs Disponibles

### Búsqueda de Línea

```
POST /api/run_line_search
Content-Type: application/json

{
    "method": "golden_section|fibonacci_search|armijo_backtracking",
    "x0": [0.0, 0.0],
    "tolerance": 1e-6,
    "max_iter": 100
}
```

### Descenso de Gradiente

```
POST /api/run_gradient_descent
Content-Type: application/json

{
    "method": "gradient_descent|gradient_descent_momentum|adam",
    "x0": [0.0, 0.0],
    "learning_rate": 0.1,
    "tolerance": 1e-6,
    "max_iter": 100
}
```

### Método de Newton

```
POST /api/run_newton
Content-Type: application/json

{
    "method": "newton_method|modified_newton|damped_newton",
    "x0": [0.0, 0.0],
    "tolerance": 1e-6,
    "max_iter": 100
}
```

## 🧪 Pruebas

### Ejecutar el ejemplo completo

```bash
python ejemplo_completo.py
```

### Probar la aplicación web

1. Ejecutar `python app.py`
2. Navegar a `http://localhost:5000`
3. Seleccionar "Búsqueda de Línea" en el menú
4. Configurar parámetros y ejecutar optimización

## 🎯 Desarrollo Futuro

### Próximas Implementaciones

- [ ] Completar interfaces web para todos los métodos
- [ ] Añadir más funciones de prueba
- [ ] Implementar comparación de métodos
- [ ] Exportar resultados a PDF/Excel
- [ ] Modo de análisis de sensibilidad
- [ ] Animaciones de convergencia

### Mejoras Técnicas

- [ ] Tests unitarios
- [ ] Documentación API con Swagger
- [ ] Base de datos para histórico
- [ ] Autenticación de usuarios
- [ ] Despliegue con Docker

## 👨‍💻 Uso Avanzado

### Agregar una nueva función objetivo

```python
def mi_funcion(x, y):
    return x**4 + y**4 - 2*x*y

def mi_gradiente(x, y):
    return np.array([4*x**3 - 2*y, 4*y**3 - 2*x])
```

### Personalizar la interfaz

- Editar `static/css/style.css` para estilos
- Modificar `templates/base.html` para layout
- Actualizar `templates/index.html` para contenido

## 🐛 Solución de Problemas

### Error: "Module not found"

```bash
pip install -r requirements.txt
```

### Error: "Port already in use"

```bash
# Cambiar puerto en app.py línea final:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Error de convergencia

- Reducir la tasa de aprendizaje
- Aumentar tolerancia
- Cambiar punto inicial

## 📞 Soporte

Para problemas o sugerencias:

1. Revisar los logs de la aplicación
2. Verificar la implementación en `metodos/`
3. Comprobar las APIs en `app.py`

---

## 🎉 ¡Listo para usar

Tu aplicación de métodos de optimización está completamente configurada y lista para ejecutar.

**Comando rápido para empezar:**

```bash
pip install -r requirements.txt && python app.py
```

Luego visita: `http://localhost:5000` 🚀
