
# LangChain Pandas DataFrame Agent

An agent that answers natural-language questions about a retail sales dataset by deciding which Pandas operations to run — tested specifically on multi-step questions (filter → group → aggregate, and nested group-then-max queries) rather than simple single-column lookups, since that's where genuine agentic reasoning is actually demonstrated.

## Why this project

My RAG and CrewAI/phidata projects demonstrate retrieval and multi-agent coordination. This project demonstrates a third, distinct agentic pattern: an agent that reasons about *which code to write* to answer a question, rather than retrieving text or coordinating with other agents. It also pairs directly with my data analyst background — this is "ask my data questions in plain English" applied to a real exploratory-analysis workflow, not just toward a toy example.

## How it works

LangChain's `create_pandas_dataframe_agent` wraps an LLM with a single tool: a live Python REPL containing the loaded DataFrame. The agent follows a ReAct (Reason → Act → Observe) loop — it writes Pandas code, the code actually executes against the real data, the result is fed back to the agent, and it decides whether it has enough information to answer or needs to run more code.

- **Dataset**: a retail sales dataset (Superstore, ~9,800 rows) with columns including Region, Category, Sub-Category, Segment, and Sales.
- **LLM**: Groq-hosted Llama 3.3 70B.
- **Interface**: Streamlit, with a dataset preview and example questions for anyone trying the demo without prior context on the data.

## A real model-size finding, not a generic claim

The first version of this project used Groq's smaller, faster `llama-3.1-8b-instant` model. It consistently failed: instead of outputting the required exact tool name (`Action: python_repl_ast`), it wrote a sentence describing the tool ("Action: Use the python_repl_ast tool to..."), which broke LangChain's ReAct output parser and caused the agent to loop on the same failed step indefinitely. Switching to the larger `llama-3.3-70b-versatile` model resolved this immediately and reliably.

This is a concrete, observed demonstration that model size affects more than answer quality — it affects whether a model can reliably follow a rigid structural output format at all, which matters specifically for ReAct-style agents that parse exact text patterns to function. A model can be fully capable of the underlying reasoning and still fail an agentic task purely on output-formatting grounds.

## Multi-step reasoning test results

Five questions were tested, increasing in reasoning complexity:

| Question                                     | Operations required                          | Result                                                                                          |
| -------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Total Sales (entire dataset)                 | Single aggregation                           | Correct (control)                                                                               |
| Average Sales by Region, Furniture only      | Filter → group → aggregate                 | Correct                                                                                         |
| Highest-Sales Sub-Category in West region    | Filter → group → aggregate → max          | Correct                                                                                         |
| Compare Consumer vs. Corporate segment Sales | Filter (2 values) → aggregate → compare    | Correct, and the agent added a percentage-difference calculation beyond what was strictly asked |
| Top-selling Sub-Category**per Region** | Group by two columns → nested per-group max | Correct                                                                                         |

The last question is the strongest evidence of genuine multi-step reasoning: finding the top Sub-Category *for each* Region requires a two-level grouped operation (group by Region and Sub-Category, sum, then take the per-Region maximum), not a single filter-and-aggregate step. The agent correctly produced `groupby(['Region', 'Sub-Category'])['Sales'].sum()` followed by a per-group `idxmax()`, and the four resulting Region/Sub-Category pairs were internally consistent with the underlying totals.

## An honest inefficiency worth reporting

On two of the five test questions, the agent wrote a correct block of Pandas code, received no observable output (because the code computed a value but never printed it), and had to re-issue the same logic with an explicit `print()` statement added before it could see a result and proceed. This cost 1–2 extra LLM calls per question without affecting final correctness — a minor but real and observable limitation in how the agent interacts with the Python REPL tool, not something to hide in favor of a cleaner-sounding success story.

## A note on `allow_dangerous_code=True`

This agent type works by having the LLM generate and execute real Python code against the DataFrame, which is a meaningfully more powerful (and riskier) pattern than a typical tool-call agent restricted to a fixed set of safe functions — LangChain requires this flag to be set explicitly, and it's treated here as a real, named tradeoff rather than boilerplate to dismiss. `langchain-experimental`, where this agent lives, is also explicitly being sunset by LangChain, which is worth knowing if extending this project later.

## What I'd improve with more time

- Add a guard or post-processing step that automatically appends `print()` to generated code blocks that compute but don't display a result, to reduce the observed iteration inefficiency.
- Test against a wider range of question phrasings (not just well-formed analytical questions) to see how the agent handles ambiguous or underspecified requests.
- Add a lightweight validation step that cross-checks the agent's numeric answers against a manually-computed Pandas equivalent, to systematically verify correctness rather than relying on manual spot-checking of each result.

## Stack

Python, LangChain (`langchain-experimental` Pandas DataFrame agent), Groq API (Llama 3.3 70B), Pandas, Streamlit
