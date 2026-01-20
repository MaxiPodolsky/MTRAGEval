import os
import json
from openai import OpenAI
from pymilvus import connections, Collection
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
import asyncio

load_dotenv()

# ==================== Configuration ====================
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')

# Map collection names to their corpus files
CORPUS_MAP = {
    "clapnq": {
        "collection": "clapnq_OpenAI_large",
    },
    "ibmcloud": {
        "collection": "cloud_OpenAI_large",
    },
    "fiqa": {
        "collection": "fiqa_OpenAI_large",
    },
    "govt": {
        "collection": "govt_OpenAI_large",
    }
}

# Input/Output files
INPUT_FILE = "../human/retrieval_tasks/taskAC.jsonl"
OUTPUT_FILE = "../results/ragtum_taskA_withinput.jsonl"

# Search parameters
TOP_K = 5  # Number of results to retrieve

# ==================== Initialize OpenAI Client ====================
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Global storage for collections
COLLECTIONS = {}


# ==================== Conversation Rewrite Functions ====================
def extract_full_conversation(input_array):
    """
    Extract full conversation with both user and agent turns.
    Returns list of tuples: (speaker, text)
    """
    conversation = []
    for turn in input_array:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        if speaker in ["user", "agent"]:
            conversation.append((speaker, text))
    return conversation


async def rewrite_query_with_context_async(conversation):
    """
    Async version of query rewriting using Gemini.
    Takes full conversation history (user + agent) and rewrites the last user turn.

    Args:
        conversation: List of (speaker, text) tuples
    """
    # Extract just user messages to check if rewriting is needed
    user_messages = [text for speaker, text in conversation if speaker == "user"]

    if len(user_messages) == 1:
        return user_messages[0]

    client = genai.Client()

    # Format the full conversation history with both user and agent
    conversation_lines = []
    for i, (speaker, text) in enumerate(conversation, 1):
        role = "User" if speaker == "user" else "Assistant"
        conversation_lines.append(f"{i}. {role}: {text}")

    conversation_text = "\n".join(conversation_lines)

    prompt = f"""
    Instructions: 
    Rewrite the LAST USER turn into a standalone, keyword-rich search query. 
    - Use the full conversation history (both user questions and assistant responses) to understand context.
    - Resolve all pronouns (it, they, that, this) using information from previous turns.
    - Remove conversational filler and polite phrases.
    - Include all relevant details and entities mentioned in the conversation.
    - If the last turn is already standalone, return it as-is.
    - Output ONLY the rewritten query text.

    Examples:

    Example 1:
    1. User: "Who is the CEO of Google?" 
    2. Assistant: "Sundar Pichai is the CEO of Google."
    3. User: "When did he take over?"
    Rewritten query: When did Sundar Pichai become CEO of Google?


    Example 2:
    1. User: "What are dialog nodes?"
    2. Assistant: "Dialog nodes are various types including the Welcome node and Anything else node. You can create custom nodes by adding a condition..."
    3. User: "What are intents?"
    4. Assistant: "Intents are the purposes or goals expressed in a customer's input, such as answering a question or processing a bill payment..."
    5. User: "How is it created?"
    Rewritten query: How is a dialog node created?

    Current Conversation:
    {conversation_text}

    Rewritten query:
    """

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )
        rewritten = response.text.strip()
        # Get last user message for logging
        last_user_msg = [text for speaker, text in conversation if speaker == "user"][-1]
        print(f"Original conversation ({len(conversation)} turns total, {len(user_messages)} user turns):")
        print(f"  Last user turn: {last_user_msg}")
        print(f"  Rewritten: {rewritten}")
        return rewritten
    except Exception as e:
        print(f"Gemini rewrite error: {e}")
        return user_messages[-1]


# ==================== Dense-Only Search Function ====================
def dense_search(query_text, corpus_key, top_k=20):
    """
    Perform dense-only search on a specific corpus collection

    Args:
        query_text: Query string
        corpus_key: Key to identify which corpus to search (e.g., 'clapnq', 'fiqa')
        top_k: Number of results to return
    """
    # Get the right collection
    if corpus_key not in COLLECTIONS:
        # Fallback logic: check if any key is a substring of the provided key
        # Useful if input is "mt-rag-clapnq..." and key is "clapnq"
        found_key = None
        for k in COLLECTIONS.keys():
            if k in corpus_key:
                found_key = k
                break

        if found_key:
            collection = COLLECTIONS[found_key]
        else:
            print(f"Error: Collection key '{corpus_key}' not found in loaded collections.")
            return []
    else:
        collection = COLLECTIONS[corpus_key]

    # Generate dense embedding using OpenAI
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query_text]
    )
    dense_q = resp.data[0].embedding

    # Perform dense search
    search_params = {"metric_type": "IP", "params": {}}

    res = collection.search(
        data=[dense_q],
        anns_field="dense_embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "text", "id"]
    )

    # Format results
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


def load_all_collections():
    """Load all Milvus collections"""
    print("\n" + "=" * 80)
    print("Loading Milvus collections...")
    print("=" * 80 + "\n")

    for corpus_key, config in CORPUS_MAP.items():
        collection_name = config["collection"]
        print(f"Loading collection: {collection_name}")

        try:
            collection = Collection(name=collection_name)
            collection.load()
            COLLECTIONS[corpus_key] = collection
            print(f"  ✓ Collection loaded successfully\n")
        except Exception as e:
            print(f"  ✗ Error loading collection: {e}\n")


# ==================== Process Queries ====================
async def process_query_batch(queries, batch_size=10):
    """Process queries in batches with concurrency limit."""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]

        # Extract full conversations
        batch_conversations = []
        for q in batch:
            if "input" in q:
                conversation = extract_full_conversation(q["input"])
                batch_conversations.append(conversation)
            else:
                batch_conversations.append([("user", q.get("text", ""))])

        # Rewrite queries in parallel
        rewrite_tasks = [
            rewrite_query_with_context_async(conversation)
            for conversation in batch_conversations
        ]
        rewritten_texts = await asyncio.gather(*rewrite_tasks, return_exceptions=True)

        # Process each query in the batch
        for j, (query_obj, query_text) in enumerate(zip(batch, rewritten_texts)):
            if isinstance(query_text, Exception):
                print(f"Error rewriting query: {query_text}")
                conv = batch_conversations[j]
                last_user = [text for speaker, text in conv if speaker == "user"]
                query_text = last_user[-1] if last_user else ""

            task_id = query_obj.get("task_id", query_obj.get("_id", "unknown"))
            collection_name = query_obj.get("Collection", "")

            if not collection_name:
                print(f"Warning: Unknown collection for task {task_id}")
                continue

            contexts = dense_search(query_text, collection_name, top_k=TOP_K)

            result_obj = query_obj.copy()
            result_obj["contexts"] = contexts
            result_obj["rewritten"] = query_text

            results.append(result_obj)

    return results


# ==================== Main Execution ====================
async def main():
    # Connect to Milvus
    print("Connecting to Milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Load all collections
    load_all_collections()

    # Load queries from mixed file
    print("\n" + "=" * 80)
    print(f"Loading queries from {INPUT_FILE}...")
    print("=" * 80 + "\n")

    queries = []
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    queries.append(obj)
        print(f"Loaded {len(queries)} queries\n")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        return

    # Process all queries
    print(f"Processing {len(queries)} queries...")
    print(f"Search parameters: Dense-only, top_k={TOP_K}\n")

    results = await process_query_batch(queries, batch_size=10)

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✓ Successfully saved {len(results)} results")

    # Display sample result
    if results:
        print("\n" + "=" * 80)
        print("Sample Result (first query):")
        print("=" * 80)
        sample = results[0]
        # Use appropriate keys based on the new structure (preserved input fields)
        task_id = sample.get('task_id', sample.get('_id', 'N/A'))
        print(f"Task ID: {task_id}")

        # Try to print the last user input from the conversation
        if 'input' in sample and isinstance(sample['input'], list):
            last_input = next((turn.get('text') for turn in reversed(sample['input']) if turn.get('speaker') == 'user'),
                              "N/A")
            print(f"Last Input: {last_input}")
        else:
            print(f"Text: {sample.get('text', 'N/A')}")

        print(f"Collection: {sample.get('Collection', 'N/A')}")
        print(f"Number of contexts: {len(sample.get('contexts', []))}")
        print(f"\nTop 3 results:")
        for i, ctx in enumerate(sample.get('contexts', [])[:3], 1):
            print(f"\n{i}. Document ID: {ctx.get('document_id')}")
            print(f"   Score: {ctx.get('score', 0):.4f}")
            print(f"   Title: {ctx.get('title')}")
            # Safely truncate text
            text_preview = ctx.get('text', '')
            print(f"   Text: {text_preview[:150]}...")
        print("=" * 80)

    # Summary
    print(f"\nSummary:")
    print(f"  Total queries processed: {len(results)}")
    if results:
        avg_results = sum(len(r.get('contexts', [])) for r in results) / len(results)
        print(f"  Average results per query: {avg_results:.1f}")


if __name__ == "__main__":
    asyncio.run(main())