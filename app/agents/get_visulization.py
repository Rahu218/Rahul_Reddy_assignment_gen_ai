import json
from textwrap import dedent
import pandas as pd
import re

from app.utils.llm import chat_with_llm2
from app.utils.run_code import run_viz_code_and_save

def generate_viz_code_from_llm(question, df, nl_summary):
    """
    Ask an LLM to generate a *simple* matplotlib Python snippet (bar / pie / line)
    based on: df.head(5), question, and nl_summary.

    Args:
        question (str): user question prompting visualization
        df (pandas.DataFrame): full dataframe (we will include head(5) snapshot in prompt)
        nl_summary (str): natural language summary for context
        openai_client: an instantiated OpenAI client object with method `chat.completions.create` or `ChatCompletion.create`.
                       Example usage below assumes openai (openai-python) with openai.ChatCompletion.create(...) or similar.
                       Adjust call to match your client.
        model (str): model name to call
        max_tokens (int), temperature (float) : LLM generation params

    Returns:
        str or None: If LLM returns runnable code, returns that code (string).
                     If LLM determines no visualization is appropriate, returns None.
    """

    # 1) Prepare a compact CSV snapshot of df.head(5)
    try:
        csv_head = df.head(5).to_csv(index=False)
    except Exception:
        # fallback to a very small human-readable table if conversion fails
        csv_head = df.head(5).to_string(index=False)

    # 2) Prompt template
    prompt = f"""
    You are an assistant that outputs a single, short, runnable Python snippet that uses only:
      - pandas (pd)
      - matplotlib.pyplot (plt)
      - io.StringIO (if needed)
    The snippet must recreate the small example dataframe below from CSV (so the code runs standalone),
    pick an appropriate chart type (bar, pie, or line) based on the question and provided NL summary,
    and save the chart to '/tmp/chart.png' (or display it with plt.show()).

    Requirements:
    1. If a visualization is NOT appropriate for the question/summary, reply exactly with the token:
       NO_VISUAL
    2. Otherwise output ONLY runnable Python code (no prose, no explanation).
    3. Keep code short and simple. Use pandas to load the CSV that follows, generate a chart using matplotlib,
       set a title, save to '/tmp/chart.png' and end.
    4. Use column names from the provided CSV. If aggregation is required, do minimal grouping (show code).
    5. Do not import heavy plotting libs; only use matplotlib and pandas.

    Here is the CSV snapshot of df.head(5):
    >>>CSV_START>>>
    {csv_head}
    <<<CSV_END<<<

    Question:
    {question}

    NL Summary:
    {nl_summary}

    Output rule reminder:
    - If no chart should be made, output exactly: NO_VISUAL
    - Otherwise output runnable Python code only.
    - the format should be like this:
    ```python
    (your code here)
    ```
    """

    response = chat_with_llm2(prompt)

    # remove code fences if present
    response = re.sub(r"```(?:python)?\n?", "", response).strip()
    response = re.sub(r"```$", "", response).strip()

    # Save the final code in a txt file for inspection
    with open("generated_viz_code.txt", "w", encoding="utf-8") as f:
        f.write(response)

    return response


# # sample
# df = pd.DataFrame({
#     "category": ["A","B","C","A","B"],
#     "value": [10, 20, 30, 5, 15],
#     "date": ["2025-01-01","2025-01-01","2025-01-02","2025-01-03","2025-01-03"]
# })

# question = "Show top categories by value"
# nl_summary = "Top categories by sum of value for the selected window."

# code_str = generate_viz_code_from_llm(question, df, nl_summary)

# if code_str is None:
#     print("No visual for this question.")
# else:
#     path = run_viz_code_and_save(code_str)
#     if path:
#         print("Chart saved as:", path)
#     else:
#         print("Failed to generate chart.")
