"""CLI helper: build the vector DB without opening the UI.

Usage:  python scripts/build_index.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import rebuild_index

if __name__ == "__main__":
    try:
        print("Starting vector index build...")
        stats = rebuild_index()
        print(f"Success! Loaded {stats['documents']} documents into {stats['chunks']} chunks in chroma_db.")
    except Exception as e:
        print(f"Error building vector index: {e}")
        sys.exit(1)

