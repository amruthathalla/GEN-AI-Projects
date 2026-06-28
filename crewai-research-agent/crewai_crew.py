from crewai import Task, Crew, Process
from crewai_agents import crewai_researcher, crewai_writer

def run_crewai(topic: str) -> str:
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'. Search the web for current, accurate "
            f"information. Find at least 3-4 distinct, important facts or developments. "
            f"For each fact, note where it came from. Flag anything that seems uncertain "
            f"or where sources disagree."
        ),
        expected_output=(
            "A list of researched facts about the topic, each with a brief note on its "
            "source, and any flagged uncertainties."
        ),
        agent=crewai_researcher,
    )

    writing_task = Task(
        description=(
            f"Using ONLY the research findings provided to you, write a clear, well-organized "
            f"summary of '{topic}' for someone with no prior background. Do not add any facts "
            f"that were not in the research. If the research flagged any uncertainty, mention "
            f"that in your summary rather than ignoring it."
        ),
        expected_output=(
            "A 200-300 word summary, organized into short paragraphs, written in plain language."
        ),
        agent=crewai_writer,
        context=[research_task],
    )

    crew = Crew(
        agents=[crewai_researcher, crewai_writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)

if __name__ == "__main__":
    topic = "Google's Gemini 2.5 Pro with Deep Think launch in June 2026"
    output = run_crewai(topic)
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(output)