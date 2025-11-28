import os
import json
import numpy as np
from openai import OpenAI
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility,
    AnnSearchRequest, WeightedRanker
)
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagAutoModel
from tqdm import tqdm

MILVUS_URI = 'http://1.92.82.153:19530'  #Milvus uri for server

### connect Milvus
connections.connect("default", host=MILVUS_URI, port="19530")

collection_name = "clapng_5000_OpenAi"  #create the collection name , name it as your like

# if already exists, drop old one
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# define Milvus schema, including dense + sparse vectors
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
schema = CollectionSchema(fields, description="Hybrid BM25 + BGE embeddings")
collection = Collection(name=collection_name, schema=schema)


### load JSONL data
file_path = "clapnq_5000.jsonl"  # load the jsonl file , here the clapnq_5000 is a test file with only 5000 docs
ids, titles, texts = [], [], []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            ids.append(obj["id"])
            titles.append(obj.get("title", ""))
            texts.append(obj.get("text", ""))

print(f"Loading {len(texts)} documents")


### load BGE model and generate dense embeddings
print("Loading BGE Model...")
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

model = FlagAutoModel.from_finetuned(
    "BAAI/bge-small-en",
    use_fp16=(device == "cuda")
)


# generate dense embeddings in batches
print("Generating BGE Embedding...")
embeddings = []
batch_size = 256  

for i in tqdm(range(0, len(texts), batch_size), desc="generate Embeddings"):
    batch = texts[i : i + batch_size]
    batch_embeddings = model.encode(batch)
    embeddings.extend(batch_embeddings.tolist())

print(f"Generated {len(embeddings)} dense vectors")


### parallel generate BM25 sparse vectors
print("Generating BM25 sparse vectors...")
tokenized_corpus = [t.lower().split() for t in tqdm(texts, desc="split text")]
bm25 = BM25Okapi(tokenized_corpus)

# parallel processing to speed up sparse vector generation
try:
    from joblib import Parallel, delayed
    
    def create_sparse_vector(doc, bm25_model):
        scores = bm25_model.get_scores(doc)
        indices = np.where(scores > 0)[0]
        values = scores[indices]
        return {int(idx): float(val) for idx, val in zip(indices, values)}
    
    print("Generating sparse vectors using parallel processing...")
    sparse_vectors = Parallel(n_jobs=-1)(
        delayed(create_sparse_vector)(doc, bm25) 
        for doc in tqdm(tokenized_corpus, desc="Genrating sparse vectors")
    )
except ImportError:
    print("Joblib is not installed, so single-threaded processing is used..")
    sparse_vectors = []
    for doc in tqdm(tokenized_corpus, desc="Generating sparse vectors"):
        scores = bm25.get_scores(doc)
        indices = np.where(scores > 0)[0]
        values = scores[indices]
        sparse_dict = {int(idx): float(val) for idx, val in zip(indices, values)}
        sparse_vectors.append(sparse_dict)

print(f"Generated {len(sparse_vectors)} sparse vectors")


### insert data into Milvus in batches
print("Batch inserting data into Milvus...")
insert_batch_size = 1000  # 每次插入 1000 条

for i in tqdm(range(0, len(ids), insert_batch_size), desc="inserting data"):
    end_idx = min(i + insert_batch_size, len(ids))
    collection.insert([
        ids[i:end_idx],
        titles[i:end_idx],
        texts[i:end_idx],
        embeddings[i:end_idx],
        sparse_vectors[i:end_idx]
    ])

# create indexes
print("Creating an index...")
collection.create_index(
    "dense_embedding", 
    {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}
)
collection.create_index(
    "sparse_embedding", 
    {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
)

# load collection into memory
collection.load()
print("Data import and indexing complete!")


#---------------------------Search Test---------------------------#


# hybrid search function
def hybrid_search(query, alpha=0.6, top_k=5):
    """    
    Args:
        query: search query string
        alpha: dense vector weight (0~1) sparse vector weight = 1 - alpha
        top_k: top k results to return
    """
    # dense embedding
    dense_q = model.encode([query])[0].tolist()
    
    # BM25 sparse embedding
    tokenized_q = query.lower().split()
    scores = bm25.get_scores(tokenized_q)
    indices = np.where(scores > 0)[0]
    values = scores[indices]
    sparse_q = {int(idx): float(val) for idx, val in zip(indices, values)}
    
    # create search requests
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
    
    # use WeightedRanker for hybrid search
    rerank = WeightedRanker(alpha, 1 - alpha)
    
    res = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=top_k,
        output_fields=["title", "text", "id"]
    )

    print(f"\n Search：{query}")
    print(f" Weights：Dense={alpha:.2f}, Sparse={1 - alpha:.2f}\n")

    for i, hit in enumerate(res[0], 1):
        print(f"{i}. [score={hit.score:.3f}] {hit.entity.get('title')}")
        print(f"   {hit.entity.get('text')[:160]}...\n")

    retrieval_json_obj = []
    contexts = []                           #format ready for eval
    for hit in res:
        context_entry = {
            "document_id": str(hit.id),
            "source": "",
            "score": hit.score,
            "text": hit.entity.get("text"),
            "title": hit.entity.get("title")
        }
        contexts.append(context_entry)
    print(contexts)


# test hybrid search
if __name__ == "__main__":
    # test queries
    queries = [
        "What happened after the French Revolution?",
        "How does machine learning work?",
        "Tell me about climate change"
    ]
    
    for query in queries:
        hybrid_search(query, alpha=0.6, top_k=5)
        print("-" * 80)
