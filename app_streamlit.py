import streamlit as st
import pandas as pd
import duckdb
import time
import os

from app.chat_session import start_session, handle_query_and_store

# summerization function
from app.summerization import get_final_summary

# visualization function
from app.agents.get_visulization import generate_viz_code_from_llm
from app.utils.run_code import run_viz_code_and_save

# -------------------------------------------------------
# Helper — Initialize DuckDB with uploaded CSV
# -------------------------------------------------------
def load_csv_to_duckdb(uploaded_file):
    df = pd.read_csv(uploaded_file)
    duckdb.sql("DROP TABLE IF EXISTS sales")
    duckdb.sql("CREATE TABLE sales AS SELECT * FROM df")
    return df


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Sales Analytics Chatbot", layout="wide")

# Session state
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "home"


# -------------------------------------------------------
# Home / Landing Page
# -------------------------------------------------------
if st.session_state.mode == "home":

    st.title("📊 Sales Analytics Assistant")
    st.markdown("""
    Welcome to your **AI-powered Sales Analytics Assistant**.  
    Upload your sales CSV and unlock:
    - 🔍 Automated KPI summarization  
    - 💬 Natural-language chat with your sales data  
    - 📈 Quick insights without writing SQL  
    """)

    uploaded = st.file_uploader("Upload Sales CSV", type=["csv"])

    if uploaded:
        df = load_csv_to_duckdb(uploaded)
        st.session_state.db_loaded = True
        st.success(f"CSV Loaded Successfully! Rows: {len(df)}")

        start_session()   # clear all vector index + conversation history

        st.write("### Next Steps")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📘 Summarize Sales Data"):
                st.session_state.mode = "summary"

        with col2:
            if st.button("💬 Chat With Data"):
                st.session_state.mode = "chat"

        st.stop()



# -------------------------------------------------------
# Summary Page
# -------------------------------------------------------
if st.session_state.mode == "summary":

    # call the summerization function
    st.title("📘 Sales Data Summary")
    with st.spinner("Generating summary..."):
        summary = get_final_summary()
        time.sleep(0.5)
    # st.markdown("### 📝 Executive Summary")
    st.markdown(summary)

    if st.button("⬅ Back to Home"):
        st.session_state.mode = "home"

    if st.button("💬 Chat With Data"):
        st.session_state.mode = "chat"



# -------------------------------------------------------
# Chat Page
# -------------------------------------------------------
if st.session_state.mode == "chat":

    st.title("💬 Chat With Your Sales Data")

    if not st.session_state.db_loaded:
        st.error("Please upload a CSV first.")
        st.stop()

    # Start new thread
    if st.button("🔄 Start New Thread"):
        st.session_state.messages = []
        start_session()
        st.toast("Started a new thread!", icon="🔄")

    # Chat UI
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    user_input = st.chat_input("Ask a question about your sales data...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Call your orchestrator flow
        with st.spinner("Thinking..."):
            res = handle_query_and_store(user_input)
            time.sleep(0.4)

        # Prepare reply text
        if res.get("status") == "ok":
            nl_summary = res.get("nl_summary", "<no summary>")
            reply_text = f"### 📝 Answer\n{nl_summary}"
        else:
            nl_summary = ""
            reply_text = f"❗ Error: {res.get('message', 'Unknown error')}"

        # Keep a textual copy in session history (so chat history remains readable)
        st.session_state.messages.append({"role": "assistant", "content": reply_text})

        # Render assistant bubble: image first, then text, then dataframe
        with st.chat_message("assistant"):
            # Then show the NL summary / answer text
            st.markdown(reply_text)


            # Generate and show visualization if applicable (try/catch so UI doesn't break)
            full_df = res.get("full_df", None)
            img_displayed = False
            if full_df is not None:
                try:
                    code_str = generate_viz_code_from_llm(user_input, full_df, nl_summary)
                    if code_str is not None:
                        path = run_viz_code_and_save(code_str)
                        if path and os.path.exists(path):
                            # Display the generated chart image first with a reasonable size
                            st.image(path, caption="Generated Chart", width=550)
                            img_displayed = True
                except Exception as e:
                    # don't break the whole UI if visualization generation fails
                    st.write("Visualization error:", str(e))

            # If no image was generated, optionally show a small placeholder or nothing
            if not img_displayed:
                # nothing here — fall through to text & table
                pass

            # Finally show the dataframe head (if provided by your orchestrator) as a scrollable table
            df_head_rows = res.get("df_head", [])
            if df_head_rows:
                try:
                    import pandas as pd
                    df_display = pd.DataFrame(df_head_rows).head(10)  # ensure max 10 rows
                    # Render scrollable dataframe (height controls scrollable region)
                    st.dataframe(df_display, height=200, width=650)
                except Exception as e:
                    st.write("Unable to render dataframe:", e)

        st.markdown("---")
        if st.button("⬅ Back to Home"):
            st.session_state.mode = "home"

