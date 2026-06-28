import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from crewai_tools import SerperDevTool

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)

search_tool = SerperDevTool()

crewai_researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, current, well-sourced information on a given topic by actively searching the web",
    backstory=(
        "You are a meticulous research analyst who never relies on memory alone. "
        "You always search for current information before reporting anything, "
        "and you note where each piece of information came from. You prioritize "
        "recent, credible sources and flag when information is uncertain or conflicting."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

crewai_writer = Agent(
    role="Technical Writer",
    goal="Turn raw research findings into a clear, well-organized summary for someone with no prior background on the topic",
    backstory=(
        "You are a technical writer who specializes in making complex topics accessible. "
        "You never invent facts that weren't in the research provided to you — you only "
        "organize, clarify, and explain what you were given. If the research is incomplete "
        "on some point, you say so rather than filling the gap yourself."
    ),
    llm=llm,
    verbose=True,
)

if __name__ == "__main__":
    print("CrewAI researcher role:", crewai_researcher.role)
    print("CrewAI writer role:", crewai_writer.role)