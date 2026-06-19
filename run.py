"""Dev/production entrypoint for the FastAPI backend.

Run with:  python run.py

Why this file exists
--------------------
`uvicorn backend.main:app --reload` (run from the project root) makes uvicorn's
file watcher recursively watch the ENTIRE working directory — which here
includes `venv/` (~72k files), `.git/`, and the large `models/` binaries. On
Windows that watcher pegs a CPU core and continually restarts the worker, and
because importing the app re-loads TensorFlow + the Keras model on every
restart, the server never settles into a state where it can answer requests —
so http://127.0.0.1:8000/docs just hangs / shows a blank page.

This launcher fixes that by scoping auto-reload to the application code only
(`backend/`) and explicitly excluding the heavy directories.

Set RELOAD=0 in the environment for a production-style run with no reloader.
"""

import os
import uvicorn

if __name__ == "__main__":
    reload_enabled = os.environ.get("RELOAD", "1") != "0"

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=reload_enabled,
        # Only watch the app source — never venv/.git/models/dataset/notebooks.
        reload_dirs=["backend"],
        reload_excludes=[
            "venv/*",
            "models/*",
            "dataset/*",
            "notebooks/*",
            ".git/*",
            "*.log",
        ],
    )
