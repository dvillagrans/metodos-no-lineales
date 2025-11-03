# 🚀 Desplegar en Render (Gratis) - 3 pasos, 5 minutos

Tu app ya está lista para desplegar. Solo sigue estos pasos:

## Paso 1: Crear cuenta en Render

1. Ve a **<https://render.com>**
2. Haz clic en "Get Started" o "Sign Up"
3. Usa tu cuenta de GitHub para registrarte (más rápido)

## Paso 2: Conectar el repositorio

1. Una vez dentro, haz clic en **"New +"** (botón azul arriba a la derecha)
2. Selecciona **"Blueprint"**
3. Conecta tu cuenta de GitHub si te lo pide
4. Busca y selecciona el repo: **`dvillagrans/metodos-no-lineales`**
5. Render detectará automáticamente el archivo `render.yaml`

## Paso 3: Desplegar

1. Haz clic en **"Apply"** o **"Deploy Blueprint"**
2. Render comenzará a:
   - Clonar tu repo
   - Construir la imagen Docker
   - Instalar dependencias (Flask, numpy, etc.)
   - Levantar el servidor con gunicorn
3. Espera 2-3 minutos
4. **¡Listo!** Tendrás una URL pública tipo:

   ```
   https://metodos-no-lineales.onrender.com
   ```

## ✅ Qué incluye el free tier de Render

- ✅ 750 horas/mes gratis
- ✅ HTTPS automático
- ✅ Logs en tiempo real
- ✅ Auto-deploy cuando haces push a GitHub
- ⚠️ El servicio "duerme" después de 15 min sin uso (tarda ~30seg en despertar)

## 🔧 Si algo falla

- Ve a la pestaña "Logs" en el dashboard de Render
- Búscame y dame el error
- Lo arreglo en minutos

---

## Alternativa: Deploy local con Docker (si quieres probarlo primero)

```powershell
# Construir la imagen
docker build -t metodos-no-lineales .

# Ejecutar el contenedor
docker run -p 5000:5000 metodos-no-lineales

# Abrir en el navegador
# http://localhost:5000
```

---

**Repo:** <https://github.com/dvillagrans/metodos-no-lineales>  
**Contacto:** Si tienes problemas, avísame y lo resuelvo al instante.
