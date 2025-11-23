import os
import time
import json
import uuid
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Create a modern OpenAI client instance (for openai-python >=1.0.0).
# If OPENAI_API_KEY is not set, the client will still attempt to read from env.
_OPENAI_CLIENT = None
def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if OPENAI_API_KEY:
            _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
        else:
            _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT

# Chroma settings - simple local in-memory by default
CHROMA_COLLECTION_NAME = "conversation_thread"
# Embedding model used for OpenAI embeddings - change if you want a different model/provider
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
# Token model for tiktoken counting. Choose a model with large context, fallback handled below.
TOKEN_COUNT_MODEL = "gpt-4o"  # used only to pick encoding; fallback occurs if not available

# Token budget from your message
DEFAULT_TOKEN_BUDGET = 16384

# default retrieval size
DEFAULT_TOP_K = 4



# ----------------------------

# Initialize chroma client and collection (simple local client)
# --- Replaces the previous chroma client initialization at top of file ---
import threading

# Lazily initialize chroma client & collection to avoid deprecated-settings error at import time
_CHROMA_LOCK = threading.Lock()
_chroma_client = None
_collection = None

def _get_chroma_client_and_collection(collection_name: str):
    """
    Lazy-initialize a chromadb client and collection.
    Uses the modern chromadb.Client() constructor (no legacy Settings passed here).
    Returns (client, collection).
    """
    global _chroma_client, _collection
    with _CHROMA_LOCK:
        if _chroma_client is not None and _collection is not None:
            return _chroma_client, _collection

        try:
            import chromadb
            # Prefer the simple constructor; this avoids legacy-settings migration issues.
            try:
                _chroma_client = chromadb.Client()
            except TypeError:
                # older versions may require Settings; fallback but minimal
                from chromadb.config import Settings
                _chroma_client = chromadb.Client(Settings())
        except Exception as e:
            raise RuntimeError("Failed to import or create chromadb client. "
                               "Install chromadb and check versions. Error: " + str(e))

        # Create or get collection safely
        try:
            # if collection exists, create_collection will raise; so check list
            existing = [c.name for c in _chroma_client.list_collections()]
            if collection_name in existing:
                _collection = _chroma_client.get_collection(collection_name)
            else:
                _collection = _chroma_client.create_collection(collection_name)
        except Exception:
            # best-effort fallback: try get_collection, then create
            try:
                _collection = _chroma_client.get_collection(collection_name)
            except Exception:
                _collection = _chroma_client.create_collection(collection_name)
        return _chroma_client, _collection


# In-memory thread log (ordered list of turn dicts). We keep this to make truncation & viewing easy.
_THREAD_LOG: List[Dict] = []


def _get_encoding(encoding_name: Optional[str] = None):
    """
    Get a tiktoken encoding. Try to use encoding_for_model, else fallback to 'cl100k_base'.
    """
    try:
        if encoding_name:
            return tiktoken.encoding_for_model(encoding_name)
    except Exception:
        pass
    # fallback
    return tiktoken.get_encoding("cl100k_base")

# token count helper
_ENCODING = _get_encoding(TOKEN_COUNT_MODEL)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text))

def _openai_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text using OpenAI embeddings API.
    Change this function if you want to use a different provider.
    """
    if OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set in environment; embeddings require an API key.")
    client = _get_openai_client()
    # New OpenAI client: client.embeddings.create(...)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    # resp.data is a list of objects with `.embedding`
    try:
        return resp.data[0].embedding
    except Exception:
        # fallback to dict-style access for different client versions
        return resp["data"][0]["embedding"]

# ----------------------------
# Thread lifecycle functions
# ----------------------------
def start_new_thread() -> None:
    global _THREAD_LOG
    _THREAD_LOG = []
    client, collection = _get_chroma_client_and_collection(CHROMA_COLLECTION_NAME)
    # Try to clear collection content without relying on deletion API across versions
    try:
        # If delete() exists on collection (newer chroma), use it
        if hasattr(collection, "delete"):
            collection.delete()
        else:
            # fallback: list and delete by ids if supported
            # attempt to get all ids and remove them
            try:
                all_ids = [m["id"] for m in collection.get(ids=None, include=["ids"]).get("ids", [])]
                if all_ids:
                    collection.delete(ids=all_ids)
            except Exception:
                # If we can't delete, attempt recreate by deleting collection then creating a new one
                try:
                    client.delete_collection(CHROMA_COLLECTION_NAME)
                    collection = client.create_collection(CHROMA_COLLECTION_NAME)
                except Exception:
                    # last resort: ignore and continue (best-effort)
                    pass
    except Exception:
        # ignore non-fatal cleanup errors
        pass


def thread_log() -> List[Dict]:
    """Return the in-memory list of turns (for viewing)."""
    return _THREAD_LOG

# ----------------------------
# Adding & storing turns
# ----------------------------
def add_turn(question: str, nl_summary: str, sql: Optional[str] = None, meta: Optional[Dict] = None) -> Dict:
    """
    Add a conversation turn to the in-memory log and the Chroma DB.
    Stores a concise textual document for RAG (question + nl_summary) and vectorizes it.

    Returns the stored turn dict.
    """
    turn_id = str(uuid.uuid4())
    ts = time.time()
    meta = meta or {}
    # canonical text to index: keep it short to reduce token usage
    doc_text = f"Q: {question}\nA: {nl_summary}"
    # Create embedding
    emb = _openai_embedding(doc_text)

    # store in chroma collection
    try:
        _collection.add(
            ids=[turn_id],
            embeddings=[emb],
            documents=[doc_text],
            metadatas=[{"question": question, "nl_summary": nl_summary, "sql": sql, "timestamp": ts, **(meta or {})}],
        )
    except Exception as e:
        # If Chroma add fails, still store in memory and continue (best-effort)
        print("Warning: failed to add to chroma collection:", e)

    turn = {
        "id": turn_id,
        "question": question,
        "nl_summary": nl_summary,
        "sql": sql,
        "meta": meta,
        "timestamp": ts,
        "doc_text": doc_text,
        "tokens": count_tokens(doc_text),
    }
    _THREAD_LOG.append(turn)
    return turn

# ----------------------------
# Context building & retrieval
# ----------------------------
def _total_tokens_for_turns(turns: List[Dict]) -> int:
    return sum(t.get("tokens", 0) for t in turns)


def get_top_k_context(question: str, k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Retrieve top-k similar historical turns from chroma using the current question as query.
    Returns list of metadata dicts with keys: id, question, nl_summary, sql, score (if available), doc_text.
    """
    # get embedding for the question
    q_emb = _openai_embedding(question)
    # query chroma
    try:
        result = _collection.query(query_embeddings=[q_emb], n_results=k, include=['metadatas', 'documents', 'distances'])
        # result structure depends on chroma version; handle common shapes
        docs = []
        if result and isinstance(result, dict):
            # older chroma returns dict with 'ids','metadatas','documents'
            metadatas = result.get("metadatas", [[]])[0]
            documents = result.get("documents", [[]])[0]
            distances = result.get("distances", [[]])[0] if "distances" in result else [None]*len(metadatas)
            for md, doc, dist in zip(metadatas, documents, distances):
                docs.append({
                    "question": md.get("question"),
                    "nl_summary": md.get("nl_summary"),
                    "sql": md.get("sql"),
                    "timestamp": md.get("timestamp"),
                    "doc_text": doc,
                    "score": dist
                })
        else:
            # sometimes returns list-like
            # best-effort parsing
            items = result[0] if isinstance(result, list) and result else result
            # fallback: return empty
            return []
        return docs
    except Exception as e:
        print("Warning: chroma query failed:", e)
        return []


def build_context_for_generation(question: str, token_budget: int = DEFAULT_TOKEN_BUDGET, top_k: int = DEFAULT_TOP_K
                                ) -> Tuple[str, List[Dict]]:
    """
    Build a single string context by selecting top-k similar turns, but respecting the token_budget.
    Strategy:
      - always include the last two turns from the in-memory thread log (if present),
      - retrieve a few candidate turns (similar to question) and include up to 2 of them (so last2 + top2),
      - include them in reverse-chronological order (newest first) until adding another would exceed budget,
      - always include the question at the end.
    Returns (context_string, included_turns_list).
    """
    candidates = get_top_k_context(question, k=top_k)  # request a few more candidates to choose from
    # map candidates to our in-memory turns to get token counts if possible
    # create small turn dicts
    candidates_turns = []
    for c in candidates:
        text = c.get("doc_text") or f"Q: {c.get('question')}\nA: {c.get('nl_summary')}"
        tokens = count_tokens(text)
        candidates_turns.append({
            "doc_text": text,
            "tokens": tokens,
            "question": c.get("question"),
            "nl_summary": c.get("nl_summary"),
            "sql": c.get("sql"),
            "score": c.get("score"),
            "timestamp": c.get("timestamp")
        })

    # sort newest first (higher timestamp)
    candidates_turns = sorted(candidates_turns, key=lambda t: t.get("timestamp", 0), reverse=True)

    # Now greedily include until token budget allows: reserve some tokens for the new question itself.
    # Reserve 2048 tokens for the question + model prompt overhead (safe margin).
    reserved_for_question =  int(0.7 * token_budget)
    allowed_for_context = token_budget - reserved_for_question

    included = []
    used = 0

    # --- NEW: always include last two turns from _THREAD_LOG (if present) ---
    # _THREAD_LOG is assumed to be ordered oldest->newest or newest last; adapt accordingly.
    # We'll take the last two entries as the most recent turns.
    last_two = []
    try:
        if len(_THREAD_LOG) >= 1:
            last_two.append(_THREAD_LOG[-1])
        if len(_THREAD_LOG) >= 2:
            last_two.append(_THREAD_LOG[-2])
    except Exception:
        last_two = []

    # Convert/normalize last_two entries to the same shape as candidates_turns and count tokens if missing
    normalized_last_two = []
    for t in last_two:
        q = t.get("question") or t.get("doc_text") or ""
        a = t.get("nl_summary") or t.get("nl") or ""
        doc_text = t.get("doc_text") or f"Q: {q}\nA: {a}"
        tokens = t.get("tokens") or count_tokens(doc_text)
        normalized_last_two.append({
            "doc_text": doc_text,
            "tokens": tokens,
            "question": q,
            "nl_summary": a,
            "sql": t.get("sql"),
            "score": t.get("score"),
            "timestamp": t.get("timestamp", 0)
        })

    # Include last_two newest-first (we want the most recent turn first)
    # normalized_last_two currently is [most recent, second most recent] because we appended -1 then -2
    for t in normalized_last_two:
        if used + t["tokens"] <= allowed_for_context:
            included.append(t)
            used += t["tokens"]
        else:
            # if even a recent turn doesn't fit, we skip it (can't force over budget)
            continue

    # --- Add up to 2 retrieved chunks (top 2) that don't duplicate last_two questions ---
    max_retrieved_to_add = 2
    added_from_candidates = 0
    seen_questions = {t.get("question") for t in included if t.get("question")}
    for t in candidates_turns:
        if added_from_candidates >= max_retrieved_to_add:
            break
        # skip if duplicate question present in last two / already included
        if t.get("question") in seen_questions:
            continue
        if used + t["tokens"] <= allowed_for_context:
            included.append(t)
            used += t["tokens"]
            added_from_candidates += 1
            if t.get("question"):
                seen_questions.add(t.get("question"))
        else:
            # skip candidate if it would exceed token budget
            continue

    # Build context string: include turns newest->oldest (we have included newest first already)
    pieces = []
    for t in included:
        pieces.append(f"Q: {t['question']}\nA: {t['nl_summary']}")
    # Append an instruction separator and the current question
    pieces.append(f"Current question: {question}")
    context_string = "\n\n".join(pieces)
    return context_string, included



# ----------------------------
# Minimal helper to view or persist thread to JSON
# ----------------------------
def dump_thread_to_json(filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_THREAD_LOG, f, default=str, indent=2)


# ----------------------------
# Example usage
# ----------------------------
# if __name__ == "__main__":
#     # Demo: start a fresh thread
#     print("Starting new thread (clearing chroma)...")
#     start_new_thread()

#     # Add a few turns (in real app you'll call add_turn after executing and summarizing)
#     add_turn("What was revenue in March 2022?", "March 2022 revenue was $94,810", sql="SELECT ...")
#     add_turn("How about April 2022?", "April 2022 revenue was $26,489,651", sql="SELECT ...")
#     add_turn("Show me May to June", "May $24,141,584; June $20,927,249", sql="SELECT ...")

#     # Now build context for a follow-up
#     q = "Give me month-over-month drop rate from April to June 2022"
#     ctx, included_turns = build_context_for_generation(q, token_budget=16384, top_k=3)
#     print("Context:\n", ctx)
#     print("Included turns:", [t["question"] for t in included_turns])
#     # Call your resolver: generate_sql(q, context=ctx)
#     # sql = generate_sql(q, context=ctx)
#     # ... then execute_sql etc.

#     # Finally dump thread to a json for inspection
#     dump_thread_to_json("thread_log.json")
#     print("Thread dumped to thread_log.json")

