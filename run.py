import os
from app import app

if __name__ == '__main__':
    # Evita problemas del reloader/watchdog en Windows/Python recientes
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), use_reloader=False)
