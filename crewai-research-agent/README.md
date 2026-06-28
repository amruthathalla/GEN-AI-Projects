
# Multi-Agent Research & Summarization: CrewAI vs. phidata

The same two-agent task — a Research Analyst searches the web, a Technical Writer summarizes the findings — implemented in two different agent frameworks (CrewAI and phidata), wrapped in a Streamlit app that lets you run either (or both, side by side) on the same topic. Built to compare how two agent frameworks structure the same underlying problem, not just to pick one and move on.

## Why this project

My other GenAI projects cover retrieval (RAG) and parameter adaptation (LoRA fine-tuning), but neither demonstrates agentic behavior — a system that decides when and how to act, rather than retrieving from a fixed knowledge base or generating from trained weights. This project fills that gap, and goes one step further: rather than building in just one agent framework, it implements the identical task in two, to produce a genuine, evidence-based comparison instead of a single framework opinion.

## How it works

Both sides follow the same two-agent design: a Research Analyst (equipped with a web search tool) gathers findings, and a Technical Writer (no search tool — works only from what the researcher provides) turns those findings into a plain-language summary. Both agents are instructed never to invent facts beyond what was actually found, and to flag uncertainty rather than presenting a guess with false confidence.

|                  | CrewAI                                                         | phidata                                                                                              |
| ---------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Search tool      | Serper (Google Search API)                                     | DuckDuckGo Search (no API key needed)                                                                |
| Task hand-off    | Built-in (`context=[research_task]`)                         | Manual — the researcher's output is passed as text into the writer's prompt by hand                 |
| Agent definition | `role` / `goal` / `backstory` (narrative)                | `role` + a list of discrete `instructions`                                                       |
| Orchestration    | `Crew` + `Process.sequential` coordinates agents and tasks | No built-in orchestration object — the calling code (`run_phidata()`) manages the sequence itself |

Both agents use the same underlying LLM (Gemini 2.5 Flash), so any difference in output reflects the framework and tool choice, not a different model.

## The comparison test, and what it actually showed

Both frameworks were tested on the same deliberately difficult topic: Google's Gemini 2.5 Pro "Deep Think" launch (June 2026) — a specific, very recent event chosen because any model's training data is unlikely to contain it, making this the clearest test of genuine retrieval versus confident-sounding generation.

**CrewAI + Serper** found real, verifiable information (the Deep Think reasoning mode, free-tier availability), but conflated two distinct, similarly-named events from different points in time — an earlier Gemini 2.5 Pro preview and the actual June 2026 launch — producing a partially accurate, partially confused summary. To its credit, the writer agent's anti-hallucination instructions worked: the final output explicitly flagged that exact dates and details were uncertain, rather than presenting the conflated information with false confidence.

**phidata + DuckDuckGo** found no usable results at all for the same query, and the writer explicitly said so — "no publicly available information," "research efforts... did not yield any results" — rather than fabricating or guessing.

**The honest conclusion**: neither framework is simply "better" based on this one test. What actually differed were the search tool's retrieval quality (Serper found more than DuckDuckGo did for this narrow, recent query) and each agent's instruction-following around uncertainty. This is exactly the value of comparing two implementations directly: it isolates that the meaningful variable here was the *search tool*, not the *orchestration framework* — a distinction that's easy to blur if you only ever build one version and assume its behavior reflects "agent frameworks" generally rather than that specific combination of choices.

## A genuine framework-design difference worth naming

CrewAI provides task-chaining as a first-class feature (`context=[research_task]`) — the framework handles passing one task's output into the next automatically. phidata has no equivalent for this kind of linear hand-off; the calling code has to manage it explicitly by formatting the researcher's output into the writer's prompt. Neither is objectively better: CrewAI's approach is more convenient for exactly this kind of fixed, linear pipeline; phidata's manual approach is more transparent and gives more control if you wanted to do something other than a straight pass-through (e.g., filtering or condensing the research before handing it to the writer).

## A note on dependency health

Both frameworks surfaced deprecation warnings during this build: phidata's Gemini integration depends on Google's now-deprecated `google.generativeai` SDK (superseded by `google.genai`), and its DuckDuckGo tool depends on a package since renamed to `ddgs`. Separately, `phidata` itself has been renamed to **Agno** in newer releases. None of this breaks current functionality, but it's worth tracking if extending this project later — these are the kind of signals worth noticing rather than ignoring, since they tend to predict where breaking changes will eventually land.

## What I'd improve with more time

- Give phidata's researcher a stronger or alternate search tool (e.g., Serper as well) to isolate whether its weaker result was due to DuckDuckGo specifically, rather than the framework.
- Run both frameworks across a larger set of topics (not just one) to see whether the date-conflation and no-results patterns observed here are consistent or one-off.
- Add a third "judge" pass that scores both outputs against the same rubric (factual accuracy, appropriate uncertainty, clarity), rather than relying on manual side-by-side reading.

## Stack

Python, CrewAI, phidata, Gemini API (free tier), Serper API, DuckDuckGo Search, Streamlit
