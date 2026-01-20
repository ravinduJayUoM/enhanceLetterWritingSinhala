"""Script to run the server from the project root."""
import os
import sys
import uvicorn

# Add rag directory to path
rag_path = os.path.join(os.path.dirname(__file__), 'rag')
sys.path.insert(0, rag_path)

if __name__ == "__main__":
    os.chdir(rag_path)  # Change to rag directory for relative path resolution
    uvicorn.run("sinhala_letter_rag:app", host="0.0.0.0", port=8000, reload=False)
