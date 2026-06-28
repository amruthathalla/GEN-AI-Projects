import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from source_text import SOURCE_TEXT

MODEL_NAME = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

def answer_extractive(question: str) -> dict:
    inputs = tokenizer(question, SOURCE_TEXT, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # The model outputs a score for every token being the START of the answer,
    # and a separate score for every token being the END of the answer.
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Confidence: how strongly the model favored this exact start/end position
    # over every other token, expressed as a 0-1 probability via softmax
    start_prob = torch.softmax(start_scores, dim=1)[0][start_index].item()
    end_prob = torch.softmax(end_scores, dim=1)[0][end_index].item()
    confidence = (start_prob + end_prob) / 2

    return {"answer": answer, "score": confidence}

if __name__ == "__main__":
    test_question = "How much does the Professional plan cost per guard?"
    result = answer_extractive(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence score: {result['score']:.4f}")