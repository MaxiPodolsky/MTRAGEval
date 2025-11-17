import dotenv
from langchain_core.tools import create_retriever_tool
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagAutoModel
from langchain_openai import OpenAIEmbeddings
import numpy as np
from pymilvus import (
    connections, Collection, AnnSearchRequest, WeightedRanker
)
import torch
import os
import json


class SimpleHybridSearch:
    def __init__(self, collection_name: str = "clapnq"):
        # Connect to Milvus
        connections.connect("default", host="127.0.0.1", port="19530")
        
        # Load collection
        self.collection = Collection(collection_name)
        self.collection.load()
        
        # Initialize BGE model
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = FlagAutoModel.from_finetuned(
        #     "BAAI/bge-small-en",
          #   use_fp16=(device == "cuda")
        # )

        dotenv.load_dotenv()

        # Initialize OpenAI embedding model
        self.model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
        
        # Initialize BM25
        self._build_local_bm25()
    
    def _build_local_bm25(self, max_docs: int = 1000):
        """Build BM25 from a subset of documents"""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = "../data/corpora/passage_level/clapnq.jsonl/clapnq.jsonl"
        
        texts = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_docs:
                        break
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            texts.append(obj.get('text', ''))
                        except Exception:
                            continue
        
        if texts:
            tokenized_corpus = [t.lower().split() for t in texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
    
    def hybrid_search(self, query: str, alpha: float = 0.6, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (dense + sparse)

        Args:
            query: Search query text
            alpha: Dense vector weights (0~1), sparse weights = 1 - alpha
            top_k: Number of results returned
        """
        # Generate dense query vectors
        # dense_q = self.model.encode([query])[0].tolist()

        dense_q = self.model.embed_query(query)
        
        # Generate sparse BM25 query vectors
        if self.bm25 is not None:
            tokenized_q = query.lower().split()
            scores = self.bm25.get_scores(tokenized_q)
            indices = np.where(scores > 0)[0]
            values = scores[indices]
            sparse_q = {int(idx): float(val) for idx, val in zip(indices, values)}
        else:
            raise RuntimeError('The local BM25 is not initialized, so a sparse query cannot be generated; please provide clapnq.json.')
        
        # Create a search request
        dense_req = AnnSearchRequest(
            data=[dense_q],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k
        )
        
        sparse_req = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k
        )
        
        # Using WeightedRanker for hybrid search
        rerank = WeightedRanker(alpha, 1 - alpha)
        
        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=rerank,
            limit=top_k,
            output_fields=["title", "text"]
        )
        
        # Formatting results
        search_results = []
        for hit in results[0]:
            search_results.append({
                "score": hit.score,
                "title": hit.entity.get("title", ""),
                "content": hit.entity.get("text", "")  # Use "content" as the name of the returned text field.
            })
            
        return search_results

# Create a search instance
searcher = SimpleHybridSearch()


def rag_retriever(query: str, k: int = 4, alpha: float = 0.6) -> List[Dict[str, Any]]:
    """
        The retrieval function used by the Agent returns a list of relevant documents.

        Args:

            query: User query text
            k: Number of documents returned
            alpha: Dense weight, sparse weight = 1 - alpha
    """
    return searcher.hybrid_search(query, alpha=alpha, top_k=k)


#My new codes：
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

class HybridRetriever(BaseRetriever):
    searcher: SimpleHybridSearch = Field(...)

    def _get_relevant_documents(self, query: str, **kwargs):
        k = kwargs.get("k", 4)
        alpha = kwargs.get("alpha", 0.6)
        docs = self.searcher.hybrid_search(query, alpha=alpha, top_k=k)
        return [
            Document(page_content=doc["content"], metadata={"score": doc["score"], "title": doc["title"]})
            for doc in docs
        ]

    def invoke(self, query, **kwargs):
        return self._get_relevant_documents(query, **kwargs)

retriever_tool = create_retriever_tool(
    HybridRetriever(searcher=searcher),
    name="hybrid_search",
    description="Search and return information about our documents, covering topics such as the French Revolution. If irrelevant, directly answer user questions using LLM."
)
