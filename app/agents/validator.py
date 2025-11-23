import re
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# -------------------------
# SQL Safety (pre-execute)
# -------------------------
SQL_FORBIDDEN_PATTERNS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bTRUNCATE\b", r"\bALTER\b", r"\bUPDATE\b",
    r"\bINSERT\s+INTO\b", r"\bINTO\s+OUTFILE\b"
]
SQL_EXPENSIVE_PATTERNS = [
    r"\bCROSS\s+JOIN\b",
    r"\bFULL\s+JOIN\b"  # sometimes expensive depending on indexes/data
]

def sql_safety_check(sql: str) -> Dict[str, Any]:
    """
    Simple safety checker for generated SQL. Should be run BEFORE executing the query.
    Returns a dict with keys:
      - safe: bool
      - issues: list[str]
    """
    issues: List[str] = []
    sql_clean = (sql or "").strip()
    if not sql_clean:
        issues.append("Empty SQL provided.")
        return {"safe": False, "issues": issues, "advice": "Provide a non-empty SQL query."}

    # Check forbidden tokens (destructive operations)
    for pat in SQL_FORBIDDEN_PATTERNS:
        if re.search(pat, sql_clean, flags=re.IGNORECASE):
            issues.append(f"forbidden operation detected: pattern `{pat}`")

    # Heuristic expensive patterns
    for pat in SQL_EXPENSIVE_PATTERNS:
        if re.search(pat, sql_clean, flags=re.IGNORECASE):
            issues.append(f"potentially expensive join pattern detected: `{pat}`")

    # Heuristic: SELECT without WHERE (possible full table scan) -> warn
    # Only warn for queries that look like selects without obvious filter
    if re.search(r"^\s*SELECT\b", sql_clean, flags=re.IGNORECASE):
        has_where = bool(re.search(r"\bWHERE\b", sql_clean, flags=re.IGNORECASE))
        has_limit = bool(re.search(r"\bLIMIT\b", sql_clean, flags=re.IGNORECASE))
        # If no WHERE and no LIMIT, warn about full table read
        if not has_where and not has_limit:
            issues.append("query has no WHERE and no LIMIT — may result in a full-table scan")

    # Semicolon at end may indicate multiple statements -> disallow for safety
    if ";" in sql_clean and not sql_clean.rstrip().endswith(";"):
        # multiple statements present
        issues.append("multiple SQL statements detected (semicolon). Only single read-only statement allowed.")
    elif sql_clean.strip().endswith(";"):
        # trailing semicolon is ok; but check inner semicolons
        if sql_clean.strip().count(";") > 1:
            issues.append("multiple SQL statements detected. Only single read-only statement allowed.")

    safe = len([i for i in issues if "forbidden" in i.lower()]) == 0
    # If there are forbidden ops, force safe = False. Otherwise, safe may be True with warnings.
    return {
        "safe": safe and True,
        "issues": issues
    }


def _parse_select_columns(sql: str) -> List[str]:
    """
    Very simple SELECT column extractor. Not a full SQL parser.
    Attempts to extract column names from the SELECT ... FROM ... clause.
    Returns list of lowercase column tokens (without aliases).
    """
    sql = sql or ""
    m = re.search(r"SELECT\s+(.*?)\s+FROM\s", sql, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    select_part = m.group(1)
    # remove parentheses content (functions that may include commas), crude but reduces noise
    select_part = re.sub(r"\([^\)]*\)", "", select_part)
    cols = [c.strip() for c in select_part.split(",")]
    # strip aliases like "col AS alias" or "col alias"
    cleaned = []
    for c in cols:
        # drop "AS alias" or " alias"
        c = re.sub(r"\s+AS\s+.*$", "", c, flags=re.IGNORECASE).strip()
        c = c.split()  # in case of "col alias"
        if c:
            cleaned.append(c[0].lower())
    # filter out '*' and empty strings
    return [c for c in cleaned if c and c != "*"]


def _extract_numbers_from_text(text: str) -> List[float]:
    """
    Find obvious numbers in the NL summary (integers or floats, optionally with commas).
    Returns floats.
    """
    if not text:
        return []
    # match numbers like 12,345.67 or 12345 or 12k (we'll ignore 'k' shorthand)
    matches = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+\.\d+\b", text.replace(",", ""))
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except Exception:
            continue
    return numbers


def validate_query_results(question: str, sql: str, df: pd.DataFrame, nl_summary: str) -> Dict[str, Any]:
    """
    Post-execution validator.
    Inputs:
      - question: original user question (string)
      - sql: generated SQL that was executed
      - df: pandas DataFrame result returned by execute_sql
      - nl_summary: Natural-language summary produced for the result (string)

    Returns dict with:
      - verdict: "pass" | "warn" | "fail"
      - confidence: float in [0,1]
      - issues: list[str]
      - suggested_fix: str or None
      - details: extra info (row_count, missing_columns, matched_claims, etc.)
    """
    issues: List[str] = []
    suggested_fix = None

    # Basic sanity
    if df is None:
        return {
            "verdict": "fail",
            "confidence": 0.0,
            "issues": ["No dataframe returned (df is None)."],
            "suggested_fix": "Ensure execute_sql returned a pandas DataFrame.",
            "details": {}
        }

    # 1) Schema & column checks: parse columns from SQL and compare to df
    parsed_cols = _parse_select_columns(sql)
    df_cols_lower = [c.lower() for c in df.columns.tolist()]
    missing_columns = []
    if parsed_cols:
        for c in parsed_cols:
            # if parsed selects are functions like COUNT(*) they might not match df columns; ignore obvious functions
            if re.search(r"\b(count|sum|min|max|avg)\b", c, flags=re.IGNORECASE):
                continue
            if c not in df_cols_lower:
                missing_columns.append(c)
        if missing_columns:
            issues.append(f"Missing columns in result: {missing_columns}")
            suggested_fix = "Verify column names or update SQL to select existing columns (check aliases)."

    # 2) Row-count sanity
    row_count = len(df)
    if row_count == 0:
        issues.append("Result has 0 rows.")
        # If question explicitly mentions a date / year, add guidance
        if re.search(r"\b(20\d{2}|19\d{2}|\b\d{4}\b)", question):
            suggested_fix = suggested_fix or "Check date filters in the query — the requested timeframe may have no data."
        else:
            suggested_fix = suggested_fix or "Check filters or try a broader timeframe."
    elif row_count > 200_000:
        issues.append(f"Very large result set ({row_count} rows). Consider adding LIMIT or more selective filters for preview.")
        suggested_fix = suggested_fix or "Add LIMIT or tighter WHERE filters for preview."

    # 3) Simple NL numeric claim validation (best-effort)
    numbers_in_nl = _extract_numbers_from_text(nl_summary)
    matched_claims = []
    nl_mismatch_issues = []
    if numbers_in_nl:
        # Compute simple aggregates to compare: sum of numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_sums = {c.lower(): float(df[c].sum()) for c in numeric_cols} if numeric_cols else {}
        for claimed in numbers_in_nl:
            # find best matching numeric_sum (closest)
            best_col = None
            best_diff = None
            for col, s in numeric_sums.items():
                diff = abs(s - claimed)
                # relative diff measure (avoid divide-by-zero)
                rel = diff / (abs(s) + 1e-9)
                if best_diff is None or rel < best_diff:
                    best_diff = rel
                    best_col = col
            if best_col is not None:
                # thresholds: exact-ish (<=2%), warn (<=5%), fail (>5%)
                if best_diff <= 0.02:
                    matched_claims.append({"claimed": claimed, "column": best_col, "rel_error": best_diff, "status": "ok"})
                elif best_diff <= 0.05:
                    nl_mismatch_issues.append(
                        f"Claim {claimed} differs from sum({best_col}) by {best_diff:.2%} (warning)."
                    )
                else:
                    nl_mismatch_issues.append(
                        f"Claim {claimed} differs from sum({best_col}) by {best_diff:.2%} (mismatch)."
                    )
            else:
                nl_mismatch_issues.append(f"Claim {claimed} could not be matched to any numeric column aggregate.")
        if nl_mismatch_issues:
            issues.extend(nl_mismatch_issues)

    # 4) Final verdict logic (simple, conservative)
    # - any 'forbidden' missing_columns is a fail? Here we treat missing columns as warn unless df is empty.
    severity_score = 0.0
    # base points for issues
    for it in issues:
        it_low = it.lower()
        if "forbidden" in it_low or "no dataframe" in it_low:
            severity_score += 2.0
        elif "0 rows" in it_low or "mismatch" in it_low or "missing columns" in it_low:
            severity_score += 1.5
        elif "very large" in it_low or "warning" in it_low:
            severity_score += 1.0
        else:
            severity_score += 0.5

    # simple confidence: fewer issues => higher confidence. Map severity to confidence
    # cap severity to something reasonable
    if severity_score == 0:
        confidence = 0.95
        verdict = "pass"
    elif severity_score < 2.5:
        confidence = max(0.6, 1.0 - severity_score * 0.2)
        verdict = "warn"
    else:
        confidence = max(0.15, 1.0 - min(severity_score, 6.0) * 0.18)
        verdict = "fail" if ("no rows" in " ".join(i.lower() for i in issues) or any("forbidden" in i.lower() for i in issues)) else "warn"

    # If there are critical missing columns AND zero rows, escalate to fail
    if missing_columns and row_count == 0:
        verdict = "fail"
        confidence = min(confidence, 0.25)
        suggested_fix = suggested_fix or "Confirm selected columns and date filters; consider regenerating SQL."

    details = {
        "row_count": row_count,
        "parsed_select_columns": parsed_cols,
        "result_columns": df.columns.tolist(),
        "missing_columns": missing_columns,
        "numeric_sums_sample": {k: v for k, v in ( {c: float(df[c].sum()) for c in df.select_dtypes(include=[np.number]).columns} ).items()},
        "numbers_found_in_nl": numbers_in_nl,
        "matched_claims": matched_claims
    }

    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 2),
        "issues": issues,
        "suggested_fix": suggested_fix,
        "details": details
    }


# # -------------------------
# # Example usage (commented)
# # -------------------------
# if __name__ == "__main__":
#     # Example (do not run DB here) - pseudo-demo
#     example_sql = "SELECT order_date, total_amount FROM sales WHERE order_date >= '2022-01-01'"
#     print(sql_safety_check(example_sql))

#     # create a tiny sample df to simulate execute_sql output
#     data = {
#         "order_date": ["2022-01-05", "2022-02-10"],
#         "total_amount": [100.0, 200.0]
#     }
#     df_sample = pd.DataFrame(data)
#     nl = "Total revenue is 300"
#     report = validate_query_results("Month-over-month revenue for 2022", example_sql, df_sample, nl)
#     print(report)