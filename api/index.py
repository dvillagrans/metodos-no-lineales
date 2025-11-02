# Vercel Python runtime entrypoint exposing the Flask WSGI app
# Docs: https://vercel.com/docs/functions/runtimes/python
import os
import sys

# Ensure the project root is on sys.path so we can import app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the Flask application object from app.py
from app import app as app  # WSGI app expected by Vercel

# Optional: set a secret key in serverless context if not present
app.config.setdefault('SECRET_KEY', os.environ.get('SECRET_KEY', 'optimization_methods_2025'))
