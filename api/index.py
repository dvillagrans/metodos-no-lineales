# Vercel Python WSGI entrypoint for Flask
# All application code is in api/ directory - app.py, metodos/, templates/, static/
from app import app

# Expose for Vercel
__all__ = ['app']
