# Thin wrapper so older docs/commands still work with Streamlit.
# Usage: streamlit run core/ui/dashboard.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import and execute the actual streamlit entry point (ui/streamlit_app.py)
try:
    import importlib
    # Try multiple module paths in order
    for module_name in ("ui.streamlit_app", "streamlit_app", "ui.app", "app"):
        try:
            app_module = importlib.import_module(module_name)
            if hasattr(app_module, "main"):
                app_module.main()
                break
            # If no main(), module execution already ran (legacy)
            break
        except ImportError:
            continue
    else:
        import streamlit as st
        st.error(f"Failed to import Streamlit app. Tried: ui.streamlit_app, streamlit_app, ui.app, app")
except Exception as e:
    import streamlit as st
    st.error(f"Error loading Streamlit app: {e}")
    raise