import streamlit as st
from pandas_agent import agent, df

st.set_page_config(page_title="Pandas DataFrame Agent", page_icon="📊", layout="wide")

st.title("📊 Ask Questions About Retail Sales Data")
st.write(
    "This agent decides which Pandas operations to run to answer your question — "
    "it can handle multi-step reasoning (filter, then group, then aggregate), not just simple lookups."
)

with st.expander("📁 Preview the dataset"):
    st.dataframe(df.head(20))
    st.caption(f"{df.shape[0]} rows · columns: {', '.join(df.columns)}")

with st.expander("💡 Try one of these example questions"):
    st.markdown("""
    - What is the total Sales across the entire dataset?
    - Among orders in the 'Furniture' category, what is the average Sales by Region?
    - Which Sub-Category has the highest total Sales in the 'West' region?
    - Compare total Sales between the 'Consumer' and 'Corporate' Segments.
    - What is the top-selling Sub-Category for each Region?
    """)

question = st.text_input("Ask a question about the data:", placeholder="e.g. Which Region has the highest total Sales?")

if st.button("Run", type="primary") and question:
    with st.spinner("Reasoning through your question..."):
        response = agent.invoke(question)
    st.markdown("### Answer")
    st.write(response["output"])