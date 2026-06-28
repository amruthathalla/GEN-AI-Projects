import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

# phidata's model wrapper for Gemini -- same underlying model as the CrewAI side,
# so any difference in output reflects the framework, not a different LLM
gemini_model = Gemini(
    id="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

phidata_researcher = Agent(
    name="Research Analyst",
    role="Find accurate, current, well-sourced information on a given topic by actively searching the web",
    model=gemini_model,
    tools=[DuckDuckGo()],
    instructions=[
        "You are a meticulous research analyst who never relies on memory alone.",
        "Always search the web for current information before reporting anything.",
        "Note where each piece of information came from.",
        "Prioritize recent, credible sources and flag when information is uncertain or conflicting.",
        "Find at least 3-4 distinct, important facts or developments on the given topic.",
    ],
    markdown=True,
)

phidata_writer = Agent(
    name="Technical Writer",
    role="Turn raw research findings into a clear, well-organized summary for someone with no prior background",
    model=gemini_model,
    instructions=[
        "You are a technical writer who specializes in making complex topics accessible.",
        "You never invent facts that weren't given to you in the research you receive.",
        "You only organize, clarify, and explain what you were given.",
        "If the research is incomplete on some point, say so rather than filling the gap yourself.",
        "Write a 200-300 word summary, organized into short paragraphs, in plain language.",
    ],
    markdown=True,
)

if __name__ == "__main__":
    print("phidata researcher role:", phidata_researcher.role)
    print("phidata writer role:", phidata_writer.role)