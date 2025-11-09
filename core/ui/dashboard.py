# Thin wrapper so older docs/commands still work with Streamlit.
# Usage: streamlit run core/ui/dashboard.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try to import the actual streamlit entry point (ui/streamlit_app.py)
try:
    from ui import streamlit_app
except ImportError as e:
    print(f"Failed to import Streamlit app: {e}")
    raise