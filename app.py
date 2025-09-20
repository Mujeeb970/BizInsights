import io
import time
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(page_title="BizInsights", page_icon="📊")
st.title("BizInsights")
st.write("Ask a question and get an answer powered by Groq.")

# Optional: user uploads a CSV (you can use it later if needed)
uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])

question = st.text_area("Your question")

# A small place to collect Q&A rows
if "qa_rows" not in st.session_state:
    st.session_state.qa_rows = []  # list of dicts

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        try:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": question}],
            )
            answer = resp.choices[0].message.content
            st.write(answer)

            # Save Q&A to session
            st.session_state.qa_rows.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": answer
            })

        except Exception as e:
            st.error(f"Error: {e}")

# If we have any Q&A rows, show a preview and a download button
if st.session_state.qa_rows:
    df = pd.DataFrame(st.session_state.qa_rows)
    st.subheader("Session log")
    st.dataframe(df, use_container_width=True)

    # Convert to CSV bytes (no index)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    fname = f"bizinsights_{int(time.time())}.csv"

    st.download_button(
        label="⬇️ Download Q&A as CSV",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv"
    )
