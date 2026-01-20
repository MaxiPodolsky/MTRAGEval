import json
import asyncio
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

# ==================== Configuration ====================
INPUT_FILE = "../results/ragtum_taskA_withinput.jsonl"
OUTPUT_FILE = "../results_C/RAG_withinput.jsonl"
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

# Safety settings to prevent 'None' responses
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
]

# Tuning Limits
MAX_CONCURRENT_ENTRIES = 10
RPM_LIMIT = 500


# ==================== Rate Limiter ====================

class RateLimiter:
    """
    Token bucket rate limiter to ensure we don't exceed RPM_LIMIT.
    """

    def __init__(self, rate_limit, period=60.0):
        self.rate_limit = rate_limit
        self.period = period
        self.tokens = rate_limit
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            new_tokens = elapsed * (self.rate_limit / self.period)
            self.tokens = min(self.rate_limit, self.tokens + new_tokens)
            self.updated_at = now

            if self.tokens >= 1:
                self.tokens -= 1
                return 0
            else:
                wait_time = (1 - self.tokens) * (self.period / self.rate_limit)
                self.tokens = 0
                self.updated_at += wait_time
                return wait_time


limiter = RateLimiter(RPM_LIMIT)


# ==================== Helper Functions ====================

async def check_relevance(client, query, context_item):
    """
    Evaluates relevance on a scale of 1-5.
    Keeps documents with a score of 2 or higher.
    Includes retry logic and model fallbacks (Fail Open if all API calls fail).
    """
    text = context_item.get('text', '')

    if not text.strip():
        return None

    # We ask for Reasoning + Score to improve model "thinking"
    prompt = f"""Evaluate the relevance of the following Document to the User Query.

Query: "{query}"
Document: "{text}"

Relevance Scale:
1: Totally Unrelated - No topical overlap.
2: Weakly Related - Mentions keywords or broad context, but no direct answer.
3: Partially Relevant - Provides context that helps understand the answer.
4: Relevant - Contains significant pieces of the answer.
5: Highly Relevant - Directly and fully addresses the query.

Instructions:
1. Provide the integer score.
2. Provide a brief one-sentence reasoning.


Response Format:
Score: <integer>
Reasoning: <brief explanation>"""

    # Retry Configuration
    MAX_RETRIES = 3

    for model_variant in MODELS_TO_TRY:
        for attempt in range(MAX_RETRIES):
            # Acquire rate limit token for every attempt
            wait_time = await limiter.acquire()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_variant,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,  # Keep deterministic
                            max_output_tokens=500,
                            safety_settings=SAFETY_SETTINGS
                        )
                    )
                )

                if not response or not response.text:
                    # If response is empty, raise exception to trigger retry
                    raise ValueError("Received empty response from API")

                response_text = response.text.strip()

                # Extract the score
                import re
                score_match = re.search(r"Score:\s*(\d)", response_text)

                if score_match:
                    score = int(score_match.group(1))
                    # THRESHOLD: Keep anything that isn't a 1 (Totally Unrelated)
                    if score >= 2:
                        return context_item
                    else:
                        return None
                else:
                    # If parsing fails, we log it and default to "Fail Open" (keeping the doc)
                    # We accept the result immediately rather than retrying to save tokens on bad formatting
                    print(f"Parse warning (no score found) from {model_variant}: {response_text[:50]}...")
                    return context_item

            except Exception as e:
                # Log the error and backoff
                print(f"Relevance check attempt {attempt + 1} failed for {model_variant}: {e}")

                # Exponential backoff: sleep 2, 4, 8 seconds...
                await asyncio.sleep(2 ** (attempt + 1))

                # If we've exhausted retries for this model, the loop moves to the next model
                if attempt == MAX_RETRIES - 1:
                    print(f"Switching from {model_variant} due to persistent errors.")
                    continue

    # Fallback: If all models and retries fail, return the item (Fail Open)
    print("All relevance checks failed. Keeping document by default.")
    return context_item


async def generate_final_answer(client, query, valid_contexts):
    if not valid_contexts:
        return "I don't know"

    context_str = "".join([f"[Document {i}]\n{ctx['text']}\n\n" for i, ctx in enumerate(valid_contexts, 1)])

    prompt = f"""You are a helpful assistant. Answer the query based on the provided documents.
- Use the information from the documents to answer the query
- If the documents contain partial information, provide that information
- Be concise but complete in your answer

Query: "{query}"

Documents:
{context_str}

Answer:"""

    # Configuration for retries
    MAX_RETRIES = 3

    for model_variant in MODELS_TO_TRY:
        for attempt in range(MAX_RETRIES):
            wait_time = await limiter.acquire()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_variant,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            safety_settings=SAFETY_SETTINGS
                        )
                    )
                )

                if response and response.text:
                    return response.text.strip()

            except Exception as e:
                # 503 is Service Unavailable; 429 is Rate Limit
                print(f"Attempt {attempt + 1} failed for {model_variant}: {e}")

                # Exponential backoff: sleep 2, 4, 8 seconds...
                await asyncio.sleep(2 ** (attempt + 1))

                # If we've exhausted retries for this model, the loop moves to the next model
                if attempt == MAX_RETRIES - 1:
                    print(f"Switching from {model_variant} due to persistent errors.")
                    continue

    return "I don't know (All models failed after retries)"


# ==================== Core Logic ====================

async def process_entry(client, entry, semaphore):
    async with semaphore:
        query = entry.get("rewritten", "")
        contexts = entry.get("contexts", [])
        task_id = entry.get("task_id", "unknown")
        input = entry.get("input", "unknown")
        collection = entry.get("Collection", "")

        # Only look at first 5 contexts
        contexts = contexts[:5]

        print(f"Processing {task_id}: {len(contexts)} contexts")

        # Filter Contexts with more lenient approach
        relevance_tasks = [check_relevance(client, query, ctx) for ctx in contexts]
        results = await asyncio.gather(*relevance_tasks)
        filtered_contexts = [ctx for ctx in results if ctx is not None]

        print(f"  → Kept {len(filtered_contexts)}/{len(contexts)} contexts")

        # Generate Answer
        final_answer = await generate_final_answer(client, query, filtered_contexts)

        return {
            "task_id": task_id,
            "query": query,
            "predictions": [{"text": final_answer}],
            "contexts": filtered_contexts,
            "input": input,
            "Collection": collection
        }


async def main():
    client = genai.Client()

    entries = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    else:
        print(f"Input file not found: {INPUT_FILE}")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ENTRIES)
    tasks = [process_entry(client, entry, semaphore) for entry in entries]

    print(f"Processing {len(entries)} entries with RPM limit {RPM_LIMIT}...")
    results = await asyncio.gather(*tasks)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
