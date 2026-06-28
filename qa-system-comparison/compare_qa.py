from extractive_qa import answer_extractive
from generative_qa import answer_generative

test_questions = [
    "How much does the Professional plan cost per guard?",          # control - already tested individually
    "How does Guard360 prevent proxy attendance?",                   # answer requires combining 2 sentences
    "What is included in the Enterprise plan?",                      # answer spans a long list, partly inherited from Professional
    "Does Guard360 require a credit card to start a free trial?",    # yes/no phrased as a question, not a direct fact lookup
    "What is Guard360's refund policy?",                              # NOT actually stated in this text -- tests honesty under missing info
]

for q in test_questions:
    print("\n" + "#" * 70)
    print(f"QUESTION: {q}")
    print("#" * 70)

    ext_result = answer_extractive(q)
    print(f"\nEXTRACTIVE: {ext_result['answer']}  (confidence: {ext_result['score']:.4f})")

    gen_result = answer_generative(q)
    print(f"GENERATIVE: {gen_result}")