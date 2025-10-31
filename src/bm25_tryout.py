import json
from pathlib import Path
from rank_bm25 import BM25Okapi


script_dir = Path(__file__).parent.parent
file_path = script_dir/'data'/'clapnq.jsonl'                #access clapnq corpus

passages = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        passages.append(json.loads(line.strip()))           #unpack corpus

#print(f"Loaded {len(passages)} passages")
#print("Example:", passages[0])

tokenized_corpus = [doc["text"].lower().split() for doc in passages]        #very simple tokenizer, just splitting words
bm25 = BM25Okapi(tokenized_corpus)

example_query = "What caused the French Revolution?"
tokenized_query = example_query.lower().split()
scores = bm25.get_scores(tokenized_query)                                   #use bm25 to get scores of passages for the query

id_map = [doc["_id"] for doc in passages]

top_k = 5
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

print("\nTop results:")
for rank, i in enumerate(top_indices, 1):
    print(f"Rank {rank}: (id={id_map[i]}), Score={scores[i]}")
    print(passages[i]["text"][:200], "...\n")

