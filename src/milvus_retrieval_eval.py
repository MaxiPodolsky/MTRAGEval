import os
import json
import math
from collections import Counter
import numpy as np
import xxhash
from openai import OpenAI
from pymilvus import (
    connections, Collection,
    AnnSearchRequest, WeightedRanker
)
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ==================== Configuration ====================
MILVUS_HOST = '1.92.82.153'
MILVUS_PORT = '19530'
COLLECTION_NAMES = [{"clapnq_OpenAI", "cloud_OpenAI", "fiqa_OpenAI"}]

# just take the one you want to test on, as far as the evaluation showed it still just evaluates one corpus, but easiest to just comment out
DATASETS = [
    {
        "collection": "clapnq_OpenAI",
        "eval_collection": "mt-rag-clapnq-elser-512-100-20240503",
        "query_file": "human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl",
        "corpus_file": "data/clapnq.jsonl"
    },
    {
        "collection": "cloud_OpenAI",
        "eval_collection": "mt-rag-ibmcloud-elser-512-100-20240502",
        "query_file": "human/retrieval_tasks/cloud/cloud_rewrite.jsonl",
        "corpus_file": "../data/cloud.jsonl"
    },
    {
        "collection": "fiqa_OpenAI",
        "eval_collection": "mt-rag-fiqa-beir-elser-512-100-20240501",
        "query_file": "human/retrieval_tasks/fiqa/fiqa_rewrite.jsonl",
        "corpus_file": "../data/fiqa.jsonl"
    }
]
# Output file path
OUTPUT_FILE = "results/retrieval_results.jsonl"

# Search parameters
ALPHA = 0.6  # Dense weight (sparse = 1 - alpha)
TOP_K = 10  # Number of results to retrieve

# BM25 parameters (matching hybrid_embedding.py)
K1 = 1.2
B = 0.75

# ==================== Initialize OpenAI Client ====================
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ==================== BM25 Functions (matching hybrid_embedding.py) ====================
def term_id(term: str) -> int:
    """Generate stable positive integer key for Milvus sparse vector"""
    return xxhash.xxh64(term).intdigest() & 0x7FFFFFFF


def compute_bm25_stats(tokenized):
    """Compute IDF and average document length for BM25"""
    N = len(tokenized)
    df = Counter()
    doc_lens = []
    for toks in tokenized:
        doc_lens.append(len(toks))
        df.update(set(toks))
    avgdl = sum(doc_lens) / max(N, 1)
    idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
    return idf, avgdl


def bm25_query_vector(tokens, idf, avgdl, k=1.2, b=0.75):
    """
    Generate BM25 sparse vector for query
    Uses the same term_id function as document indexing
    """
    tf = Counter(tokens)
    # For queries, we use a simplified scoring (no document length normalization)
    vec = {}
    for t, f in tf.items():
        if t in idf:
            # Query term weight
            w = idf[t] * f
            if w > 0:
                vec[term_id(t)] = float(w)
    return vec



# ==================== Hybrid Search Function ====================
def hybrid_search(query_text, alpha=0.6, top_k=10):
    """
    Perform hybrid search combining OpenAI dense embeddings and BM25 sparse embeddings
    Matches the embedding approach from hybrid_embedding.py

    Args:
        query_text: Query string
        alpha: Weight for dense embedding (0-1), sparse weight = 1 - alpha
        top_k: Number of results to return

    Returns:
        List of context dictionaries for evaluation
    """
    # Generate dense embedding using OpenAI (matching hybrid_embedding.py)
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    )
    dense_q = resp.data[0].embedding

    # Generate BM25 sparse embedding (matching hybrid_embedding.py approach)
    sparse_q = {}
    if IDF:  # Only if we have BM25 stats
        tokenized_q = query_text.lower().split()
        sparse_q = bm25_query_vector(tokenized_q, IDF, AVGDL, k=K1, b=B)

    # Create search requests
    dense_search_params = {"metric_type": "IP", "params": {}}
    sparse_search_params = {"metric_type": "IP", "params": {}}

    dense_req = AnnSearchRequest(
        data=[dense_q],
        anns_field="dense_embedding",
        param=dense_search_params,
        limit=top_k
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_q],
        anns_field="sparse_embedding",
        param=sparse_search_params,
        limit=top_k
    )

    # Perform hybrid search with weighted ranker
    rerank = WeightedRanker(alpha, 1 - alpha)

    res = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=top_k,
        output_fields=["title", "text", "id"]
    )

    # Format results for evaluation
    contexts = []
    for hit in res[0]:
        context_entry = {
            "document_id": str(hit.entity.get('id')),
            "source": "",
            "score": float(hit.score),
            "text": hit.entity.get("text", ""),
            "title": hit.entity.get("title", "")
        }
        contexts.append(context_entry)

    return contexts

print("Connecting to Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

open(OUTPUT_FILE, "w").close()              #clear the file once of old results SAVE BEFORE RUNNING NEW!

for dataset in DATASETS:
    COLLECTION_NAME = dataset["collection"]
    EVAL_COLLECTION_NAME = dataset["eval_collection"]
    QUERY_FILE = dataset["query_file"]
    CORPUS_FILE = dataset["corpus_file"]

    print("\n============================================")
    print(f"Running dataset: {COLLECTION_NAME}")
    print("============================================\n")
    # ==================== load connection from milvus ====================

    # Load collection
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    print(f"Collection '{COLLECTION_NAME}' loaded successfully")

    # ==================== Load Corpus for BM25 Stats ====================
    print("Loading corpus for BM25 statistics...")

    corpus_texts = []
    IDF = {}
    AVGDL = 0

    try:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    text = (obj.get("content") or obj.get("text") or "").strip()
                    corpus_texts.append(text)

        print(f"Loaded {len(corpus_texts)} documents")
        print("Computing BM25 statistics...")
        tokenized_corpus = [t.lower().split() for t in tqdm(corpus_texts, desc="Tokenizing")]
        IDF, AVGDL = compute_bm25_stats(tokenized_corpus)
        print(f"Computed IDF for {len(IDF)} unique terms, AVGDL={AVGDL:.2f}")

    except FileNotFoundError:
        print(f"Warning: Corpus file '{CORPUS_FILE}' not found.")
        print("BM25 sparse search will be disabled.")

    # ==================== Load Queries ====================
    print(f"\nLoading queries from {QUERY_FILE}...")
    queries = []
    try:
        with open(QUERY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    queries.append(obj)
        print(f"Loaded {len(queries)} queries")
    except FileNotFoundError:
        print(f"Warning: Query file '{QUERY_FILE}' not found!")

    # ==================== Process All Queries ====================

    print(f"\nProcessing {len(queries)} queries...")
    print(f"Search parameters: alpha={ALPHA}, top_k={TOP_K}")
    print(f"BM25 parameters: k1={K1}, b={B}")
    print(f"Output file: {OUTPUT_FILE}\n")

    results = []
    for query_obj in tqdm(queries, desc="Processing queries"):
        task_id = query_obj.get("_id", "unknown")
        query_text = query_obj.get("text", "")

        if not query_text:
            print(f"Warning: Empty query for task_id={task_id}, skipping...")
            continue

        # Perform hybrid search
        contexts = hybrid_search(query_text, alpha=ALPHA, top_k=TOP_K)

        # Create result object in evaluation format
        result_obj = {
            "task_id": task_id,
            "query": query_text,
            "contexts": contexts,
            "Collection": EVAL_COLLECTION_NAME
        }

        results.append(result_obj)

    # ==================== Save Results ====================
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Successfully saved {len(results)} results")
    print(f"\nResults ready for evaluation!")

# ==================== Display Sample Results ====================
if results:
    print("\n" + "=" * 80)
    print("Sample Result (first query):")
    print("=" * 80)
    sample = results[0]
    print(f"Query ID: {sample['task_id']}")
    print(f"Query: {sample['query']}")
    print(f"Collection: {sample['Collection']}")
    print(f"Number of contexts: {len(sample['contexts'])}")
    print(f"\nTop 3 results:")
    for i, ctx in enumerate(sample['contexts'][:3], 1):
        print(f"\n{i}. Document ID: {ctx['document_id']}")
        print(f"   Score: {ctx['score']:.4f}")
        print(f"   Title: {ctx['title']}")
        print(f"   Text: {ctx['text'][:150]}...")
    print("=" * 80)

# ==================== Summary Statistics ====================
print(f"\nSummary:")
print(f"  Total queries processed: {len(results)}")
if results:
    avg_results = sum(len(r['contexts']) for r in results) / len(results)
    print(f"  Average results per query: {avg_results:.1f}")
    print(f"  Output format: Evaluation-ready JSONL")