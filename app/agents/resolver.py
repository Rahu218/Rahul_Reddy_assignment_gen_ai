import os, re, json
from typing import Optional
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.agents.validator import sql_safety_check

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY

MODEL = os.getenv("RESOLVER_MODEL", "gpt-4o-mini")

SYSTEM_TEMPLATE = """You are a SQL generator for DuckDB for a single table named `Data`.
The table `sales` has the following columns:

- 'Date' (str): Order or transaction date (In YYYY-MM-DD format).
- 'Style' (str): Internal product style code (base model).
- 'Category' (str): High-level product category from your catalog.
- 'Size' (str): SKU size such as('S', '3XL', 'XL', 'L', 'XXL')
- 'Courier Status' (str): Shipment status (Has 3 unique values: 'Shipped', 'Cancelled', 'Unshipped').
- 'Qty' (int64): Quantity purchased in the order.
- 'Amount' (float64): Order amount payable by customer before fees.
- 'Revenue'(float64): Total Revenue generated due to this order.
- 'ship-city' (str): Destination city for delivery.
- 'ship-state' (str): Destination state for delivery.
- 'B2B' (bool): Whether the order was a business (GST invoice) order.(True/False)
- 'fulfilled-by' (str): Whether the order was fulfilled by Easy Ship, Merchant, or Amazon (FBA).


Instructions:
- Output ONLY one SELECT statement (DuckDB dialect). No explanation.
- Allowed clauses: SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT.
- For question asking for Month over Month Growth, write query showing change in 'Amount' and 'Qty'.
- Do NOT use SUBSTRING() on date columns in DuckDB. DuckDB does NOT support SUBSTRING(Date,...). Always use strftime(Date, '%Y-%m') or cast the date to VARCHAR before applying substring
- Never produce DDL/DML (CREATE, DROP, INSERT, UPDATE, DELETE, etc).
- Do not invent column names not in the available columns.
"""

FEW_SHOT = """
NL: Top 5 product categories by revenue for the month of '2022-04'
SQL: SELECT "Category", SUM("Amount") AS total_revenue FROM sales
 WHERE "Date" >= '2022-04-01'
   AND "Date" <  '2022-05-01'
 GROUP BY "Category"
 ORDER BY total_revenue DESC
 LIMIT 5

NL: List all the months we have data for
SQL: SELECT DISTINCT 
    strftime("Date", '%Y-%m') AS month
FROM sales
ORDER BY month;


NL: Top 10 cities by average order amount for Amazon-fulfilled orders (min 5 orders)
SQL: SELECT "ship-city", AVG("Amount") AS avg_amount, COUNT(*) AS orders
 FROM sales
 WHERE "fulfilled-by" = 'Amazon (FBA)'
 GROUP BY "ship-city"
 HAVING COUNT(*) >= 5
 ORDER BY avg_amount DESC
 LIMIT 10

NL: Month-over-month revenue for Category 'Apparel' for the year '2022'
SQL: SELECT 
    strftime("Date", '%Y-%m') AS month,
    SUM("Amount") AS revenue
FROM sales
WHERE "Category" = 'Apparel'
  AND "Date" >= DATE '2022-01-01'
  AND "Date" <  DATE '2023-01-01'
GROUP BY month
ORDER BY month;
"""

PROMPT_TMPL = """{system_instructions}

Examples:
{few_shot_examples}

User question:
{question}

Respond with ONLY the SQL SELECT statement or CLARIFY: ...
"""
def sanitize_model_sql(sql_text: str) -> str:
    # strip code fence
    sql = re.sub(r"```(?:sql)?\n?", "", sql_text).strip()
    sql = re.sub(r"```$", "", sql)
    # remove leading "SQL:" etc
    sql = re.sub(r"^SQL:\s*", "", sql, flags=re.IGNORECASE).strip()
    # remove trailing semicolons
    sql = sql.rstrip().rstrip(";")
    return sql

def generate_sql(question: str, context: Optional[str] = None) -> str:
    # Prepare the prompt template (do not format into a plain string)
    # The template must expose these variables: system_instructions, few_shot_examples, question
    prompt_template = PromptTemplate(
        template=PROMPT_TMPL, 
        input_variables=["system_instructions", "few_shot_examples", "question"]
    )

    # If you have context, prefix it when forming the system_instructions or pass it separately.
    # Here we'll prepend context to the question text (alternatively adjust as you prefer).
    if context:
        # You can either embed context into the question variable or prepend to the template
        question_with_context = f"CONTEXT: {context}\n\n{question}"
    else:
        question_with_context = question

    # Model runnable
    model = ChatOpenAI(model=MODEL, temperature=0.7, max_tokens=4096)

    # Compose runnables into a runnable sequence
    chain = prompt_template | model | StrOutputParser()

    # Invoke the chain with the variables the PromptTemplate expects
    resp = chain.invoke({
        "system_instructions": SYSTEM_TEMPLATE,   # or resolved system string
        "few_shot_examples": FEW_SHOT,
        "question": question_with_context
    })

    sql = sanitize_model_sql(resp)

    # safety check
    safety = sql_safety_check(sql)
    if not safety["safe"]:
        issues = "; ".join(safety["issues"])
        raise RuntimeError(f"Generated SQL failed safety check: {issues}")
    print(safety["issues"])

    return sql
