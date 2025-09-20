import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="BizInsights", page_icon="📊", layout="centered")

st.title("BizInsights")
st.write("Ask a question and get insights.")

# (Optional) upload a file if your original app used one
uploaded_file = st.file_uploader("Upload a CSV (optional)", type=["csv"])

prompt = st.text_area("Your question")

if st.button("Ask"):
    if not prompt.strip():
        st.warning("Please type a question.")
    else:
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            msg = [{"role": "user", "content": prompt}]
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msg,
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
