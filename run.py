import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app
from backend.config import HOST, PORT

if __name__ == '__main__':
    print("=" * 65)
    print(" GLUCOMETRIC AI: DIABETES PREDICTION SYSTEM ")
    print(f" Web Server live at: http://{HOST}:{PORT}")
    print(" Press Ctrl+C to terminate.")
    print("=" * 65)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
