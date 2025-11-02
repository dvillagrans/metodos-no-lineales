# Vercel Python WSGI entrypoint for Flask
# All application code is in api/ directory - app.py, metodos/, templates/, static/
import sys
import os

try:
    from app import app
    # Expose for Vercel
    __all__ = ['app']
except Exception as e:
    # Fallback debug app if main app fails to import
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    @app.route('/<path:path>')
    def debug_error(path=''):
        import traceback
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'cwd': os.getcwd(),
            'sys_path': sys.path,
            'files_in_cwd': os.listdir('.'),
            'files_in_api': os.listdir('api') if os.path.exists('api') else 'no api dir',
            'python_version': sys.version
        }), 500
