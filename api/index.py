# Vercel Python runtime entrypoint exposing the Flask WSGI app
# Docs: https://vercel.com/docs/functions/runtimes/python

# Vercel expects the WSGI app to be named 'app' or exposed as the module
# We import it from the parent directory where app.py lives
import sys
from pathlib import Path

# Add parent directory to path to import app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the Flask application - Vercel will use this as the WSGI app
from app import app
