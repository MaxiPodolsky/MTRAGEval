import dotenv
from pymilvus import connections
from tools.retriever_tools import HybridRetriever, searcher
from dotenv import load_dotenv

dotenv.load_dotenv()

def main():
    # Connect to Milvus
    connections.connect("default", host="127.0.0.1", port="19530")

    # Initialize your retriever (uses the same searcher you defined)
    retriever = HybridRetriever(searcher=searcher)

    # Example queries
    queries = [
        "What happened after the French Revolution?",
        "How do deployments work in Kubernetes?",
        "Define matrix multiplication",
    ]

    for q in queries:
        print(f"\n=== QUERY: {q}")
        docs = retriever.invoke(q, k=5, alpha=0.6)  # alpha = dense weight
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("title", "")
            score = d.metadata.get("score", 0)
            snippet = (d.page_content or "").replace("\n", " ")[:180]
            print(f"{i}. [score={score:.4f}] {title}")
            print("   ", snippet, "...\n")

if __name__ == "__main__":
    main()
