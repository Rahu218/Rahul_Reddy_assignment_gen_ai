# README / Technical Notes

## 📦 Setup & Execution Guide

1.  **Clone the repository** or download the ZIP file and open it in VS Code or your preferred development environment.
2.  **Install dependencies** listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```
3.  **Set your OpenAI API key** inside the `.env` file:

    ```
    OPENAI_API_KEY=your_key_here
    ```
4.  **Launch the application UI** by running:

    ```bash
    python app_streamlit.py
    ```

    This starts the **Streamlit interface** where you can upload CSV files and interact with the chat-based analytics agent.

---

## 📌 Assumptions & Data Schema Notes

* The system is designed around the **Amazon Sales Report** CSV schema.
* All prompts, SQL generation logic, and validation rules are built assuming this structure.
* An additional computed column named **`Revenue`** is added programmatically:

    ```python
    df['Revenue'] = np.where(df['Courier Status'] == 'Shipped',
                             df['Amount'] * df['Qty'],
                             0)
    ```

* Any new CSV used in the system **must follow the same schema** to ensure smooth SQL generation and analysis.

---

## ⚠️ Limitations

* The **validation agent** is implemented at a basic level and does not perform deep SQL quality checks.
* As a result, the **confidence score** may not always be accurate.
* The current version uses only the **first 30 rows** of the output DataFrame when generating natural language insights due to token constraints.
    This limits the scalability of more complex queries.

---

## 🚀 Possible Improvements

* Enhance the **validation agent** with more robust SQL correctness checks and rule-based schema validation.
* Improve the **execution agent**, especially the part where it combines:
    * User question
    * Generated SQL output
    * Resulting DataFrame
    to produce richer insights.
* Replace the *first-30-rows* limitation with a smarter summarization or chunking strategy for larger result sets.
* Add better error-handling and schema-flexible parsing to support more CSV file types.

---

## 📁 Project Structure
Blend360_GenAI_assignment.
|   .env
|   app_streamlit.py
|   explore.ipynb
|   folder_structure.txt
|   README.md
|   requirement.txt
|   
+---app
|   |   chat_session.py
|   |   conversation_manager.py
|   |   final_summary.txt
|   |   orchestrator.py
|   |   summerization.py
|   |   summerization.txt
|   |   __init__.py
|   |   
|   +---agents
|   |       executor.py
|   |       get_visulization.py
|   |       resolver.py
|   |       validator.py
|   |       __init__.py
|   |       
|   +---data
|   |       Data.csv
|   |       
|   \---utils
|           llm.py
|           run_code.py
|           
\---logs
        chatbot.log