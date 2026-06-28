from pandas_agent import agent

test_questions = [
    "What is the total Sales across the entire dataset?",                                    # single-step, sanity check
    "Among orders in the 'Furniture' category, what is the average Sales by Region?",         # filter + group + aggregate
    "Which Sub-Category has the highest total Sales in the 'West' region?",                   # filter + group + aggregate + sort
    "Compare total Sales between the 'Consumer' and 'Corporate' Segments.",                    # filter (multiple values) + group + aggregate
    "What is the top-selling Sub-Category for each Region?",                                   # group + nested aggregation
]

for q in test_questions:
    print("\n" + "#" * 70)
    print(f"QUESTION: {q}")
    print("#" * 70)
    response = agent.invoke(q)
    print("\nFINAL ANSWER:", response["output"])