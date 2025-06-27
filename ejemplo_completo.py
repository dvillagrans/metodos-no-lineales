"""
Ejemplos de uso de todos los métodos de optimización implementados
"""

import numpy as np
import matplotlib.pyplot as plt
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


def ejemplo_funcion_objetivo(x, y):
    """Función de prueba: f(x,y) = (x-1)² + (y-2)²"""
    return (x - 1)**2 + (y - 2)**2


def ejemplo_gradiente(x, y):
    """Gradiente de la función de prueba"""
    return np.array([2*(x - 1), 2*(y - 2)])


def ejemplo_hessiana(x, y):
    """Hessiana de la función de prueba"""
    return np.array([[2, 0], [0, 2]])


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


def main():
    print("🔧 DEMOSTRACIÓN DE MÉTODOS DE OPTIMIZACIÓN NO LINEAL")
    print("=" * 60)
    
    # Punto inicial
    x0 = np.array([0.0, 0.0])
    
    print(f"📍 Punto inicial: {x0}")
    print(f"📊 Función objetivo: f(x,y) = (x-1)² + (y-2)²")
    print(f"🎯 Óptimo esperado: x* = [1, 2], f* = 0")
    print()
    
    # ✅ 1. BÚSQUEDA DE LÍNEA (SECCIÓN ÁUREA)
    print("1️⃣ BÚSQUEDA DE LÍNEA - SECCIÓN ÁUREA")
    print("-" * 40)
    
    line_search = LineSearchOptimizer()
    result1 = line_search.golden_section(
        ejemplo_funcion_objetivo, ejemplo_gradiente, x0,
        max_iter=100, tol=1e-6
    )
    
    print(f"✅ Resultado: x* = {result1['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result1['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones: {result1['iterations']}")
    print(f"✔️ Convergió: {result1['converged']}")
    print()
    
    # ✅ 2. DESCENSO DE GRADIENTE
    print("2️⃣ DESCENSO DE GRADIENTE")
    print("-" * 40)
    
    grad_desc = GradientDescentOptimizer()
    result2 = grad_desc.gradient_descent(
        ejemplo_funcion_objetivo, ejemplo_gradiente, x0,
        learning_rate=0.1, max_iter=100, tol=1e-6
    )
    
    print(f"✅ Resultado: x* = {result2['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result2['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones: {result2['iterations']}")
    print(f"✔️ Convergió: {result2['converged']}")
    print()
    
    # ✅ 3. MÉTODO DE NEWTON
    print("3️⃣ MÉTODO DE NEWTON (CON HESSIANA)")
    print("-" * 40)
    
    newton = NewtonOptimizer()
    result3 = newton.newton_method(
        ejemplo_funcion_objetivo, ejemplo_gradiente, ejemplo_hessiana, x0,
        max_iter=100, tol=1e-6
    )
    
    print(f"✅ Resultado: x* = {result3['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result3['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones: {result3['iterations']}")
    print(f"✔️ Convergió: {result3['converged']}")
    print()
    
    # ✅ 4. MÉTODOS PRIMALES (GRADIENTE RESTRINGIDO)
    print("4️⃣ MÉTODOS PRIMALES - GRADIENTE PROYECTADO")
    print("-" * 40)
    
    constrained = ConstrainedOptimizer()
    result4 = constrained.projected_gradient(
        ejemplo_funcion_objetivo, ejemplo_gradiente,
        constraints_eq=[restriccion_igualdad],
        constraints_eq_jac=[restriccion_igualdad_jac],
        x0=np.array([1.0, 1.0]),  # Punto factible para x + y = 2
        learning_rate=0.1, max_iter=100, tol=1e-6
    )
    
    print(f"✅ Resultado: x* = {result4['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result4['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones: {result4['iterations']}")
    print(f"✔️ Convergió: {result4['converged']}")
    print(f"⚖️ Violación restricciones: {result4['violations'][-1]:.6f}")
    print()
    
    # ✅ 5. MÉTODOS DE PENALIZACIÓN
    print("5️⃣ MÉTODOS DE PENALIZACIÓN EXTERIOR")
    print("-" * 40)
    
    penalty = PenaltyBarrierOptimizer()
    result5 = penalty.exterior_penalty(
        ejemplo_funcion_objetivo, ejemplo_gradiente,
        constraints_eq=[restriccion_igualdad],
        x0=x0, penalty_start=1.0, max_outer_iter=10, max_inner_iter=50
    )
    
    print(f"✅ Resultado: x* = {result5['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result5['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones externas: {result5['outer_iterations']}")
    print(f"✔️ Convergió: {result5['converged']}")
    print(f"⚖️ Violación restricciones: {result5['violations'][-1]:.6f}")
    print()
    
    # ✅ 6. MÉTODO DE BARRERA LOGARÍTMICA
    print("6️⃣ MÉTODO DE BARRERA LOGARÍTMICA")
    print("-" * 40)
    
    try:
        # Punto factible para x² + y² ≤ 4
        x0_factible = np.array([0.5, 0.5])
        result6 = penalty.logarithmic_barrier(
            ejemplo_funcion_objetivo, ejemplo_gradiente,
            constraints_ineq=[restriccion_desigualdad],
            x0=x0_factible, barrier_start=1.0, max_outer_iter=10
        )
        
        print(f"✅ Resultado: x* = {result6['x']:.4f}")
        print(f"📈 Valor óptimo: f* = {result6['errors'][-1]:.6f}")
        print(f"🔄 Iteraciones externas: {result6['outer_iterations']}")
        print(f"✔️ Convergió: {result6['converged']}")
        print()
    except Exception as e:
        print(f"⚠️ Error en barrera logarítmica: {e}")
        print()
    
    # ✅ 7. MULTIPLICADORES DE LAGRANGE
    print("7️⃣ MULTIPLICADORES DE LAGRANGE (NEWTON-KKT)")
    print("-" * 40)
    
    lagrange = LagrangeMultiplierOptimizer()
    result7 = lagrange.lagrange_newton(
        ejemplo_funcion_objetivo, ejemplo_gradiente, ejemplo_hessiana,
        constraints_eq=[restriccion_igualdad],
        constraints_eq_jac=[restriccion_igualdad_jac],
        x0=np.array([1.0, 1.0]), max_iter=50
    )
    
    print(f"✅ Resultado: x* = {result7['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result7['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones: {result7['iterations']}")
    print(f"✔️ Convergió: {result7['converged']}")
    print(f"🎛️ Multiplicadores: λ = {result7['lambda']:.4f}")
    print(f"⚖️ Violación KKT: {result7['violations'][-1]:.6f}")
    print()
    
    # ✅ 8. LAGRANGIANO AUMENTADO
    print("8️⃣ LAGRANGIANO AUMENTADO")
    print("-" * 40)
    
    result8 = lagrange.augmented_lagrangian(
        ejemplo_funcion_objetivo, ejemplo_gradiente,
        constraints_eq=[restriccion_igualdad],
        constraints_eq_jac=[restriccion_igualdad_jac],
        x0=x0, max_outer_iter=10, max_inner_iter=50
    )
    
    print(f"✅ Resultado: x* = {result8['x']:.4f}")
    print(f"📈 Valor óptimo: f* = {result8['errors'][-1]:.6f}")
    print(f"🔄 Iteraciones externas: {result8['outer_iterations']}")
    print(f"✔️ Convergió: {result8['converged']}")
    print(f"🎛️ Multiplicadores: λ = {result8['lambda_eq']:.4f}")
    print(f"⚖️ Violación restricciones: {result8['violations'][-1]:.6f}")
    print()
    
    print("🎉 ¡TODOS LOS MÉTODOS COMPLETADOS!")
    print("=" * 60)
    print("📚 RESUMEN DE MÉTODOS IMPLEMENTADOS:")
    print("✅ 1. Búsqueda de línea (sección áurea)")
    print("✅ 2. Descenso de gradiente")
    print("✅ 3. Método de Newton (con hessiana)")
    print("✅ 4. Métodos primales (gradiente restringido)")
    print("✅ 5. Métodos de penalización exterior")
    print("✅ 6. Métodos de barrera (logarítmica, inversa)")
    print("✅ 7. Multiplicadores de Lagrange (Newton-KKT)")
    print("✅ 8. Lagrangiano aumentado")
    print()
    print("🔥 ¡Tu colección de métodos de optimización está COMPLETA!")


if __name__ == "__main__":
    main()
