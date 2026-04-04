"""Script to run the server from the project root."""
import os
import sys
import uvicorn

# Add rag directory to path
rag_path = os.path.join(os.path.dirname(__file__), 'rag')
sys.path.insert(0, rag_path)

# Load .env from rag/ directory manually
_env_path = os.path.join(rag_path, '.env')
if os.path.exists(_env_path):
    try:
        with open(_env_path, encoding='utf-8-sig') as _f:
            for _line in _f:
                _line = _line.strip().replace('\x00', '')
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception as _e:
        print(f"[Warning] Could not load .env file: {_e}")

if __name__ == "__main__":
    os.chdir(rag_path)  # Change to rag directory for relative path resolution
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
