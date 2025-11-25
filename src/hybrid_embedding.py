import os
import json
import math
from collections import Counter
import numpy as np
import xxhash
from openai import OpenAI
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# 1) Connect Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# 2) Use a consistent collection name
collection_name = "clapnq"

# 3) Drop/recreate
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# 4) Schema (dense + sparse)
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
schema = CollectionSchema(fields, description="Hybrid BM25 + OpenAI embeddings")
collection = Collection(name=collection_name, schema=schema)

# ---------- Load JSONL ----------
JSONL_PATH = "../data/corpora/passage_level/clapnq.jsonl/clapnq.jsonl"

ids, titles, texts = [], [], []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        ids.append(str(obj.get("id") or obj.get("document_id")))
        titles.append(obj.get("title", "") or "")
        texts.append((obj.get("content") or obj.get("text") or "").strip())

print(f"Loaded {len(texts)} documents from {JSONL_PATH}")

# ---------- Dense embeddings ----------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embeddings = []
batch_size = 256
for i in tqdm(range(0, len(texts), batch_size), desc="Generating dense embeddings"):
    batch = texts[i: i + batch_size]
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
    embeddings.extend([item.embedding for item in resp.data])
print(f"Generated {len(embeddings)} dense vectors")

# ---------- BM25 sparse embeddings ----------
print("Computing BM25 sparse embeddings")

tokenized_corpus = [t.lower().split() for t in tqdm(texts, desc="Tokenizing")]

def term_id(term: str) -> int:
    # stable positive integer key for Milvus
    return xxhash.xxh64(term).intdigest() & 0x7FFFFFFF

def compute_bm25_stats(tokenized):
    N = len(tokenized)
    df = Counter()
    doc_lens = []
    for toks in tokenized:
        doc_lens.append(len(toks))
        df.update(set(toks))
    avgdl = sum(doc_lens) / max(N, 1)
    idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
    return idf, avgdl

IDF, AVGDL = compute_bm25_stats(tokenized_corpus)

def bm25_doc_vector(tokens, idf, avgdl, k=1.2, b=0.75):
    tf = Counter(tokens)
    dl = len(tokens)
    K = k * (1 - b + b * (dl / max(avgdl, 1e-9)))
    vec = {}
    for t, f in tf.items():
        w = idf.get(t, 0.0) * ((f * (k + 1)) / (f + K))
        if w > 0:
            vec[term_id(t)] = float(w)
    return vec

sparse_vectors = [
    bm25_doc_vector(tokens, IDF, AVGDL, k=1.2, b=0.75)
    for tokens in tqdm(tokenized_corpus, desc="Generating BM25 doc vectors")
]

print(f"Generated {len(sparse_vectors)} sparse vectors")

# ---------- Insert into Milvus ----------
insert_batch_size = 1000
for i in tqdm(range(0, len(ids), insert_batch_size), desc="Inserting data"):
    end = min(i + insert_batch_size, len(ids))
    collection.insert([
        ids[i:end], titles[i:end], texts[i:end],
        embeddings[i:end], sparse_vectors[i:end]
    ])

# ---------- Indexing ----------
collection.create_index(
    "dense_embedding",
    {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}
)
collection.create_index(
    "sparse_embedding",
    {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
)
collection.load()

print("Data import and indexing complete!")
