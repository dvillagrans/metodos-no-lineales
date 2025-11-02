# Vercel Python WSGI entrypoint for Flask
# The parent directory is automatically added to sys.path by Vercel
import sys
import os

# Vercel sets working directory to project root, but we need to ensure imports work
# Add the project root explicitly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the Flask app - Vercel will use this as the WSGI application
try:
    from app import app
except ImportError as e:
    # Fallback: create a minimal error-reporting Flask app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return jsonify({
            'error': 'Import failed',
            'details': str(e),
            'sys_path': sys.path,
            'cwd': os.getcwd(),
            'files': os.listdir('.')
        }), 500
