#!/usr/bin/env python
"""Seed demo: register a demo user and upload the sample document.

Usage (from project root, with venv active):
    python scripts/seed_demo.py
"""
import os
import sys
from pathlib import Path

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEMO_USER = os.getenv("DEMO_USERNAME", "demo")
DEMO_PASS = os.getenv("DEMO_PASSWORD", "demo1234")
SAMPLE_DOC = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "ml_basics_notes.txt"


def main():
    print(f"=== DocuQuery Seed Demo (target: {BASE_URL}) ===\n")

    # 1. Register
    print(f"[1] Registering user '{DEMO_USER}' ...")
    r = requests.post(f"{BASE_URL}/api/auth/register", json={"username": DEMO_USER, "password": DEMO_PASS})
    if r.status_code == 201:
        token = r.json()["access_token"]
        print(f"    Registered OK. Token acquired.")
    elif r.status_code == 400 and "already" in r.json().get("error", ""):
        print(f"    User already exists, logging in ...")
        r2 = requests.post(f"{BASE_URL}/api/auth/login", json={"username": DEMO_USER, "password": DEMO_PASS})
        r2.raise_for_status()
        token = r2.json()["access_token"]
        print(f"    Logged in OK. Token acquired.")
    else:
        print(f"    ERROR: {r.status_code} {r.text}")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload sample doc
    print(f"\n[2] Uploading '{SAMPLE_DOC.name}' ...")
    if not SAMPLE_DOC.exists():
        print(f"    Sample doc not found at {SAMPLE_DOC}. Skipping upload.")
    else:
        with open(SAMPLE_DOC, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/documents",
                files={"file": (SAMPLE_DOC.name, f, "text/plain")},
                headers=headers,
            )
        if r.status_code == 201:
            doc = r.json()
            print(f"    Uploaded. doc_id={doc['id']} status={doc['status']}")
            doc_id = doc["id"]
        else:
            print(f"    ERROR: {r.status_code} {r.text}")
            sys.exit(1)

    # 3. Ask a question (first call — LLM)
    print(f"\n[3] Asking a question (first call — LLM) ...")
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"question": "What is supervised learning?", "document_id": doc_id, "top_k": 4},
        headers=headers,
    )
    if r.status_code == 200:
        d = r.json()
        print(f"    cache_hit={d['cache_hit']}  provider={d['provider']}  model={d['model']}")
        print(f"    Answer: {d['answer'][:200]}")
    else:
        print(f"    ERROR: {r.status_code} {r.text}")

    # 4. Same question again — cache hit
    print(f"\n[4] Same question again — expect cache_hit=true ...")
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"question": "What is supervised learning?", "document_id": doc_id, "top_k": 4},
        headers=headers,
    )
    if r.status_code == 200:
        d = r.json()
        print(f"    cache_hit={d['cache_hit']}  (should be True)")
        assert d["cache_hit"] is True, "Expected cache hit!"
        print(f"    Answer: {d['answer'][:200]}")
    else:
        print(f"    ERROR: {r.status_code} {r.text}")

    print("\n=== Seed demo complete ===")


if __name__ == "__main__":
    main()
