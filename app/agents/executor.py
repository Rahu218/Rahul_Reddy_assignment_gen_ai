import time, re
import duckdb
import pandas as pd
from typing import Dict, Any, Optional
import sqlglot

from app.utils.llm import chat_with_llm2


def detect_used_columns(sql: str, schema_cols: Optional[list] = None):
    # crude: extract token-like words that match schema column names
    if not schema_cols:
        return []
    found = set()
    for c in schema_cols:
        pattern = r"\b" + re.escape(c) + r"\b"
        if re.search(pattern, sql, flags=re.IGNORECASE):
            found.add(c)
    return list(found)

# Fuction to convert dataframe to text (the full table as text)
def df_to_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    # Limit to max_rows for brevity
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_string(index=False)

def NL_summarize_results(df, colums, question):
    # Prompt to summarize results
    prompt = f"""
    Generate a concise natural-language Answer to the question: "{question}" as per the results in the data table below.
    - The answer should be based ONLY on the data provided and only answer what is asked.
    - Provide specific values from the data where applicable.
    - Provide any additional insights that can be inferred from the data.
    - The output should be only the final Answer, without any preamble or explanation.

    The columns used: {', '.join(colums)}

    Given the following data table:
        {df_to_text(df)}
    """
    response = chat_with_llm2(prompt)

    return response.strip()


def execute_sql(sql: str, csv_path: str = "C:/Users/91939/Downloads/Blend360 assignment/app/data/Data.csv", schema_cols: Optional[list]=None) -> Dict[str, Any]:

    # Create an in-memory DuckDB connection, register CSV as table for convenience
    con = duckdb.connect(database=':memory:')
    # register CSV as a table named sales (if not too big)
    con.execute(f"CREATE TABLE sales AS SELECT * FROM read_csv_auto('{csv_path}')")

    t0 = time.time()
    try:
        df = con.execute(sql).fetchdf()
    except Exception as e:
        raise RuntimeError(f"Execution error: {e}")
    elapsed = (time.time() - t0) * 1000.0

    # sample meta: null rates for used columns
    used_cols = detect_used_columns(sql, schema_cols)
    null_rates = {}
    for c in used_cols:
        if c in df.columns:
            null_rates[c] = int(df[c].isna().sum()) / max(len(df), 1)
    
    # Generate Natural-language summarization of results
    nl_answer = NL_summarize_results(df, used_cols, sql)

    return {
        "df": df,
        "row_count": len(df),
        "meta": {
            "nl_answer": nl_answer,
            "elapsed_ms": elapsed,
            "sample_rows": min(200, len(df)),
            "used_columns": used_cols,
            "null_rates": null_rates
        }
    }
