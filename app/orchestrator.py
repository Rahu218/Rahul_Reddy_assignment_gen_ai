# orchestrator.py
import logging
from typing import Any, Dict, Optional

# Import your agents - adjust paths as needed
# from blend360_assignment.app.agents.resolver import generate_sql
from app.agents.resolver import generate_sql
from app.agents.executor import execute_sql
from app.agents.validator import sql_safety_check, validate_query_results

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")


def orchestrate_question(question: str, context = None) -> Dict[str, Any]:
    try:
        # 1) Generate SQL
        sql = generate_sql(question, context=context)
        logger.info("Generated SQL: %s", sql)

        # 2) Pre-execution safety check
        safety = sql_safety_check(sql)

        if not safety.get("safe", False):
            # Block destructive queries. Return an actionable error for the UI to show.
            logger.warning("SQL safety check failed: %s", safety["issues"])
            return {
                "status": "blocked",
                "reason": "sql_safety_failed",
                "safety_issues": safety["issues"],
                "sql": sql,
            }

        # 3) Execute SQL (your executor)
        exec_result = execute_sql(sql)
        if not isinstance(exec_result, dict) or "df" not in exec_result:
            logger.error("execute_sql returned unexpected result: %s", exec_result)
            return {
                "status": "error",
                "reason": "executor_error",
                "message": "execute_sql did not return expected {'df', 'meta'} structure.",
                "raw": exec_result,
            }

        df = exec_result["df"]
        meta = exec_result["meta"]

        # 4) Obtain NL summary: prefer executor-provided summary otherwise fallback
        nl_summary = meta["nl_answer"]
        if not nl_summary:
            logger.info("No NL summary from executor")
            nl_summary = "No summary available."
            return {
                "status": "error",
                "reason": "no_nl_summary",
                "message": "Executor did not provide a natural language summary.",
                "sql": sql,
            }

        # 5) Validate results (log but do not block further)
        validation_report = validate_query_results(question, sql, df, nl_summary)
        logger.info("Validation report: %s", validation_report)

        # # head of df print
        # print(df.head(10))

        # 6) Build final response object; include everything useful for UI / debugging
        response = {
            "status": "ok",
            "question": question,
            "sql": sql,
            "df_head": df.head(10).to_dict(orient="records") if df is not None else [],
            "row_count": len(df) if df is not None else 0,
            "nl_summary": nl_summary,
            # "validation": validation_report,
            "executor_meta": meta,
            'full_df': df
        }
        return response

    except Exception as e:
        logger.exception("Orchestration failed: %s", e)
        return {
            "status": "error",
            "reason": "exception",
            "message": str(e),
        }


# # -------------------------
# # If you want to use it as a tiny CLI/test harness
# # -------------------------
# if __name__ == "__main__":
#     # quick demo — replace with real questions in your app
#     test_q = "Month-over-month revenue for 2022"
#     out = orchestrate_question(test_q, verbose=False)
#     import json
#     print(json.dumps(out, indent=2))
