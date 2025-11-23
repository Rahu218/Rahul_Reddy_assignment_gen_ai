# chat_session.py
import os
import inspect
import json
import logging
from typing import Any, Dict, Optional

# Make sure logs folder exists
os.makedirs("logs", exist_ok=True)

# Configure logging to file instead of terminal
logging.basicConfig(
    filename="logs/chatbot.log",         # log file path
    filemode="a",                        # append mode
    level=logging.INFO,                  # log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("chat_session_simple")

# import only the small set of functions from your modules (keeps things simple)
from app.conversation_manager import (
    start_new_thread,
    build_context_for_generation,
    add_turn,
    dump_thread_to_json,
    thread_log,
)
from app.orchestrator import orchestrate_question  # expects to orchestrate a single query; we prefer it to accept context

# Config
TOKEN_BUDGET = 16384
TOP_K = 3



def start_session() -> None:
    """Convenience wrapper to start a new thread/session."""
    logger.info("Starting new thread/session (clears conversation_manager index/log).")
    start_new_thread()


def handle_query_and_store(question: str) -> Dict[str, Any]:
    """
    Build RAG context, call orchestrator (with context if supported), store the resulting turn in conversation DB.
    Returns the orchestrator result.
    """
    # 1) Build context (RAG)
    context, included_turns = build_context_for_generation(question, token_budget=TOKEN_BUDGET, top_k=TOP_K)
    logger.info("Context built. Included turns: %s", [t["question"] for t in included_turns])

    # 2) Call orchestrator (which does generate_sql -> safety -> execute -> summarize -> validate)
    # We pass context if orchestrator supports it.
    try:
        result = orchestrate_question(question, context=context)
    except Exception as e:
        # If anything goes wrong, return an error-shaped dict
        logger.exception("Orchestrator call failed: %s", e)
        return {"status": "error", "message": f"Orchestrator error: {e}"}

    # 3) If orchestrator returned ok, persist the turn to the RAG index using add_turn
    if isinstance(result, dict) and result.get("status") == "ok":
        nl_summary = result.get("nl_summary") or result.get("exec_meta", {}).get("nl_answer") or "Auto-summary not available."
        sql = result.get("sql")
        try:
            stored = add_turn(question, nl_summary, sql=sql, meta={"row_count": result.get("row_count", 0)})
            logger.debug("Turn stored: %s", stored.get("id") if stored else "store failed")
        except Exception as e:
            logger.exception("Failed to add turn to conversation index: %s", e)
    else:
        logger.info("Orchestrator returned non-ok status; not adding to RAG index.")

    # 4) return the orchestrator result for display/testing
    return result


def cli_loop():
    print("Simple chat CLI (single thread). Commands: /new (new thread), /dump (dump thread to JSON), /exit")
    start_session()
    while True:
        try:
            q = input("\nYou: ").strip()
            if not q:
                continue
            if q.lower() in ("/exit", "/quit"):
                print("bye")
                break
            if q.lower() == "/new":
                start_session()
                print("Started new thread.")
                continue
            if q.lower() == "/dump":
                dump_thread_to_json("thread_dump.json")
                print("Thread dumped to thread_dump.json")
                continue

            res = handle_query_and_store(q)
            print("\n== RESPONSE ==")
            # Simple, helpful printing
            if not isinstance(res, dict):
                print("Unexpected response type:", type(res), res)
                continue

            status = res.get("status")
            print("Status:", status)
            if status != "ok":
                # print whole result for debugging
                print(json.dumps(res, indent=2, default=str))
                continue

            print("NL Summary:\n", res.get("nl_summary", "<no summary>"))
            print("\nSQL:\n", res.get("sql", "<no sql>"))
            print("\nRow count:", res.get("row_count", 0))
            print("\nSample rows (head):")
            for r in res.get("df_head", []):
                print(r)
            # print("\nValidation:", json.dumps(res.get("validation", {}), indent=2, default=str))

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            logger.exception("Error in CLI loop: %s", e)
            print("Error:", e)


if __name__ == "__main__":
    cli_loop()
