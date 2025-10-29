"""
retrieval.py
------------
Implements retrieval functions for the MT-RAG benchmark dataset.

Dataset Attribution:
This code uses data from the MT-RAG (Multilingual Text Retrieval-Augmented Generation) Benchmark
developed by IBM Research.

Source:
https://github.com/IBM/mt-rag-benchmark

License:
Apache License 2.0
© IBM Research, 2024
"""


import os, json, gzip, zipfile
from pathlib import Path
from typing import List, Dict, Optional

def load_passages_from_zip(zip_path: str | Path, limit: Optional[int] = None) -> List[Dict]:
    zip_path = Path(zip_path)
    assert zip_path.suffix == ".zip", f"Expected a .zip file, got {zip_path}"
    inner_filename = zip_path.stem  # file.jsonl.zip → file.jsonl

    passages = []
    n = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner_filename) as file:
            for raw_line in file:
                record = json.loads(raw_line)
                text = (record.get("text") or "").strip()
                if not text:
                    continue
                passages.append({
                    "id": record.get("document_id") or record.get("id"),
                    "title": record.get("title", ""),
                    "text": text
                })
                n += 1
                if limit and n >= limit:
                    break
    return passages