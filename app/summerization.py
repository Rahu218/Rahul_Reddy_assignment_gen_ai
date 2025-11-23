from app.orchestrator import orchestrate_question
from app.utils.llm import chat_with_llm2



def get_final_summary():
    level_0 = [
        "What is the total number of orders, total revenue of the dataset?",
        "What date range does the data span (start date to end date)?",
        "How many unique Categories, Styles, Sizes, and ship-states are present?"
    ]

    level_1 = [
        "What is the total Revenue, total Quantity sold, and total Amount collected?",
        "What is the month-over-month revenue trend for 2022 year?",
        # "What is the Year on Year revenue growth?", # only 2022 data is present
        "What is the overall cancellation rate (percentage of orders cancelled or unshipped)?",
        "What is the average order value (AOV = Revenue / total orders)?",
        "What is the B2B vs B2C split in terms of order count and revenue?"
    ]

    level_2 = [
        # Category Level
        "Which Category generated the highest Revenue?",
        "What is the revenue share percentage of each Category?",
        "Which Category grew or declined the most? on month-over-month basis for 2022?",
        
        # Geography Level
        "What are the top ship-states by Revenue?",
        "Top 5 states contributed the most to growth in terms of Revenue for 2022 year?",
        "Top 5 states contributed the least to growth in terms of Revenue for 2022 year?",
        
        # Fulfillment
        "What percentage of orders were fulfilled by Easy Ship, Merchant, and FBA?",
        "Which fulfillment method has the highest cancellation or unshipped rate?",
        
        # B2B Segment
        "What is the B2B contribution to total revenue and orders?",
        "Are B2B orders growing or shrinking over the different months in 2022?" # complex
    ]

    level_3 = [
        "What are the best-performing Styles or SKUs by Revenue? Give me top 5",
        "What are the worst-performing SKUs with declining sales? Give me bottom 5",
        "Which Sizes sell the most? give me top 5",
        "Which Sizes sell the least? give me top 5",
        "Are any specific Sizes over-represented in cancellations?" # complex
    ]

    level_4 = [
        "What percentage of orders were Shipped, Cancelled, and Unshipped?",
        "Which states have operational issues such as higher cancellation or unshipped rates? Give me top 5"
    ]

    level_5 = [
        "What is the monthly revenue trend for the entire dataset period?",
        "Which month had the highest revenue and which had the lowest in Year 2022?"
    ]

    level_7 = [
        "What were the top categories behind revenue growth the full period?",
        "What fulfilment issues contributed to cancellations?",
        "Which products drove most of the overall performance (positive or negative)?"
    ]

    level_8 = [
        "Which States should be prioritized for growth in Revenue? so give me states with declining revenue trends", # complex
        "What operational improvements (fulfillment methods) can help reduce cancellations?",
        "Which SKUs should be stocked up based on Qty orders?",
    ]

    all_questions = level_0 + level_1 + level_2 + level_3 + level_4 + level_5 + level_7 + level_8

    # context = ""

    # for question in all_questions:
    #     responce = orchestrate_question(question)
    #     nl_answer = responce.get("nl_summary", "No summary available.")
    #     # add question and responce to context
    #     context += f"Q: {question}\nA: {nl_answer}\n\n"
    
    # create a txt file and copy past the context
    # with open("app/summerization.txt", "w", encoding="utf-8") as f:
    #     f.write(context)

    # read the context from the file
    with open("app/summerization.txt", "r", encoding="utf-8") as f:
        context = f.read()
    # print(context)

    prompt = f"""
        You are a senior analytics consultant. You will receive a set of Q&A insights extracted from a sales dataset. 
        Each insight is written in the form:

        Q: <question>
        A: <natural language answer>

        Your task is to synthesize ALL the Q&A pairs into a clear, concise, and professional sales performance summary.

        Follow these rules strictly:

        1. **Do not repeat the Q&A pairs.**
        2. **Summarize only the insights, not the questions.**
        3. Structure the final output as a clean executive report with the following sections:

        ---

        ## Executive Summary
        Provide 3–5 sentences highlighting:
        - Overall revenue performance  
        - Growth or decline trends  
        - High-level demand patterns  
        - Key wins and challenges

        ## Key Performance Metrics
        Provide bullet points for:
        - Total revenue, quantity sold, and AOV  
        - MoM or YoY trends  
        - Cancellation and fulfillment performance  
        - B2B vs B2C split  
        - Any major anomalies or risk items

        ## Category & Product Insights
        Summarize:
        - Best/worst performing categories  
        - Top SKUs and their contribution  
        - Size-level insights  
        - Underperforming or declining product groups

        ## Geography & Fulfillment Performance
        Highlight:
        - Top revenue-generating states/cities  
        - Regions contributing to growth or decline  
        - Fulfillment channel performance  
        - States/cities with high cancellations

        ## Operational & Risk Insights
        Mention:
        - Cancellation/Unshipped impact  
        - Revenue at risk  
        - Any spikes, anomalies, or operational bottlenecks

        ## Recommendations
        Give 3–5 actionable, strategic recommendations:
        - Inventory decisions  
        - Marketing priorities  
        - Fulfillment improvements  
        - High-risk areas to monitor  
        - Opportunities to grow revenue

        ---

        ### IMPORTANT:
        - Focus on clarity and professionalism.
        - Use numbers only if provided in the answers.
        - If conflicting data appears, choose the most consistent trend.
        - Do not invent any values — use only what appears in the answers.

        ---

        Now here is the dataset Q&A context:
        {context}
        """

    final_prompt = prompt
    final_summary = chat_with_llm2(final_prompt)

    # if '---' in final_summary, remove '---'
    if '---' in final_summary:
        final_summary = final_summary.replace('---', '')

    # save final summary to a text file
    with open("app/final_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)

    return final_summary

# get_final_summary()
