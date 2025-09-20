import streamlit as st
from groq import Groq

st.set_page_config(page_title="BizInsights", page_icon="📊")
st.title("BizInsights")
st.write("Ask a question and get an answer powered by Groq.")

# Optional: upload a CSV if you plan to parse data later
uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])

question = st.text_area("Your question")

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
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
