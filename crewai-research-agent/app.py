import streamlit as st
from crewai_crew import run_crewai
from phidata_crew import run_phidata

st.set_page_config(page_title="Multi-Agent Research: CrewAI vs phidata", page_icon="🔬", layout="wide")

st.title("🔬 Multi-Agent Research & Summarization")
st.write(
    "The same two-agent task — a Research Analyst searches the web, a Technical Writer "
    "summarizes the findings — implemented in two different agent frameworks. "
    "Pick a framework (or run both) and compare how each handles the same topic."
)

with st.expander("ℹ️ About this comparison"):
    st.markdown("""
    - **CrewAI** uses Serper (Google Search API) as its search tool, and built-in task-chaining
      (`context=[research_task]`) to pass the researcher's findings to the writer automatically.
    - **phidata** uses DuckDuckGo Search (no API key needed), and findings are passed between
      agents manually — phidata has no built-in task-chaining for this kind of linear pipeline.
    - Both agents share the same underlying LLM (Gemini 2.5 Flash) and the same role instructions,
      so any difference in output reflects the framework/tool, not a different model.
    """)

topic = st.text_input("Enter a topic to research:", placeholder="e.g. recent developments in quantum computing")

framework = st.radio(
    "Which framework should run this?",
    ["CrewAI", "phidata", "Both (side by side)"],
    horizontal=True,
)

if st.button("Run", type="primary") and topic:
    if framework == "CrewAI":
        with st.spinner("Researching and writing with CrewAI..."):
            result = run_crewai(topic)
        st.markdown("### CrewAI Result")
        st.write(result)

    elif framework == "phidata":
        with st.spinner("Researching and writing with phidata..."):
            result = run_phidata(topic)
        st.markdown("### phidata Result")
        st.write(result)

    else:  # Both, side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### CrewAI Result")
            with st.spinner("Running CrewAI..."):
                crewai_result = run_crewai(topic)
            st.write(crewai_result)
        with col2:
            st.markdown("### phidata Result")
            with st.spinner("Running phidata..."):
                phidata_result = run_phidata(topic)
            st.write(phidata_result)