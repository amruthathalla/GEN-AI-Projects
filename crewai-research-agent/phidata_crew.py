from phidata_agents import phidata_researcher, phidata_writer

def run_phidata(topic: str) -> str:
    # Step 1: researcher runs first, using its search tool
    research_prompt = (
        f"Research the topic: '{topic}'. Search the web for current, accurate "
        f"information. Find at least 3-4 distinct, important facts or developments. "
        f"For each fact, note where it came from. Flag anything that seems uncertain "
        f"or where sources disagree."
    )
    research_response = phidata_researcher.run(research_prompt)
    research_findings = research_response.content

    # Step 2: writer runs second, receiving the researcher's findings as input text
    # (this is the manual equivalent of CrewAI's `context=[research_task]` --
    # phidata has no built-in task-chaining, so we pass the prior output ourselves)
    writing_prompt = (
        f"Here are research findings on '{topic}':\n\n{research_findings}\n\n"
        f"Using ONLY the research findings above, write a clear, well-organized "
        f"summary for someone with no prior background. Do not add any facts that "
        f"were not in the research. If the research flagged any uncertainty, mention "
        f"that in your summary rather than ignoring it."
    )
    writing_response = phidata_writer.run(writing_prompt)
    return writing_response.content

if __name__ == "__main__":
    topic = "Google's Gemini 2.5 Pro with Deep Think launch in June 2026"
    output = run_phidata(topic)
    print("\n" + "=" * 60)
    print("FINAL OUTPUT (phidata)")
    print("=" * 60)
    print(output)