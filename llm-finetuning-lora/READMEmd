# LoRA Fine-Tuning: Teaching a Small LLM a Fixed Answer Format

A lightweight fine-tuning project that uses LoRA (Low-Rank Adaptation) to teach `distilgpt2`, a small (82M-parameter) open-source language model, to consistently answer questions in a fixed structure:

```
<answer>
Confidence: high
```

## Why this project

After building a RAG system that retrieves external information for a model to use, I wanted to understand the other major way of adapting LLM behavior: changing the model's own parameters rather than what it's given at inference time. Fine-tuning a small model on a free GPU (Google Colab) with LoRA was the most realistic way to do that hands-on, on a short timeline.

## How it works

1. **Base model**: `distilgpt2`, loaded via Hugging Face `transformers`.
2. **LoRA setup** (`peft` library): instead of updating all ~82 million of the model's parameters, LoRA freezes the original model and trains a small additional set of parameters (~295,000, about 0.36% of the total) attached to the model's attention layers. This is what makes fine-tuning feasible on free hardware.
3. **Dataset**: 10 hand-written question/answer pairs, each phrased two ways, for 20 small training examples — built to demonstrate the technique on a tight budget, not to teach the model broad new knowledge.
4. **Label masking**: training labels are masked (`-100`) over the question portion of each example, so the model is only graded on how well it predicts the *answer*, not on reproducing the question.
5. **Training**: 20 epochs, learning rate 2e-4, on a free Colab T4 GPU.
6. **Evaluation**: a held-out, untouched copy of the base model is compared against the fine-tuned model on the same questions, using deterministic generation (`do_sample=False`) so the comparison is fair and repeatable.

## Results

Training loss dropped from approximately 5.2 to below 0.1 over training — clear evidence the model was fitting the target pattern.

**Seen example** (in training data):

| | Output |
|---|---|
| Question | What is the capital of Japan? |
| Base model | *rambling, unrelated text about Japanese government policy* |
| Fine-tuned model | `Tokyo` / `Confidence: high` |

**Unseen example** (Germany was never in the training data):

| | Output |
|---|---|
| Question | What is the capital of Germany? |
| Base model | *rambling, unrelated text about German government policy* |
| Fine-tuned model | `Berlin` / `Confidence: high` *(format correct; first line accurate, generation degrades afterward)* |

**A failure case, reported honestly**:

| | Output |
|---|---|
| Question | Who wrote Romeo and Juliet? |
| Base model | *rambling, unrelated text* |
| Fine-tuned model | Format pattern attempted, but factual answer was incorrect, and output degraded into unrelated text after the first line |

## What this actually demonstrates (and what it doesn't)

- **The model reliably learned the target *format*** — the shift from "no structure at all" to "starts with an answer-like line, often followed by a confidence line" is consistent and visible across both seen and unseen questions.
- **It partially generalized**, correctly answering "Germany" → "Berlin" despite never seeing that question during fine-tuning — most likely because the base model's pretraining already contained that fact, and fine-tuning taught it the *format* to express it in.
- **It did not reliably learn new facts**, and generation quality degrades after the first line in several cases. With only 20 training examples on a 35-epoch (later reduced to 20-epoch) run with an 82M-parameter model, this is an expected limitation, not a flaw in the method — it illustrates the real difference between teaching a model a response *shape* versus teaching it *content*, which is a genuine and useful distinction in fine-tuning research.

## What I'd improve with more time

- Use a larger and more varied training set (more facts, more phrasings) to test whether the failure mode (factual errors, post-first-line rambling) is a data-scale problem or a model-scale problem.
- Try a slightly larger base model to see whether format adherence and factual reliability both improve.
- Add a quantitative scoring pass (e.g., does the output match the target format via regex, across N held-out questions) rather than relying on manual inspection of a few examples.

## Stack

Python, Hugging Face `transformers`, `peft` (LoRA), `datasets`, Google Colab (free T4 GPU)