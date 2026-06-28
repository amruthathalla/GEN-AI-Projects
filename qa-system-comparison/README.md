
# Extractive vs. Generative Question Answering

A controlled comparison of two QA approaches on the same source text and questions: extractive QA (a BERT-family model trained on SQuAD, which highlights an exact answer span from the source) versus generative QA (an LLM that writes an answer in its own words, grounded in the same source via prompting). Built to compare the two mechanisms directly, not to demonstrate a single "best" approach.

## Why this project

My RAG projects already use grounded generation. This project isolates a different, more fundamental question: what does extractive QA — which can never hallucinate a fact not in the text, because it can only point at existing spans — actually buy you over generation, and what does it cost? Testing both on the same text and the same questions turns that question into something measurable rather than asserted.

## Setup

- **Source text**: cleaned product/feature copy from a real company's marketing site (Guard360, a security guard management platform) — note this is product page text, not a dedicated FAQ page, since the source site had no separate FAQ section.
- **Extractive model**: `distilbert-base-cased-distilled-squad`, used via direct model/tokenizer calls (the high-level `pipeline("question-answering")` API was unavailable in the installed `transformers` version, so start/end logits were computed and decoded manually).
- **Generative model**: Llama 3.3 70B via Groq, using the same "answer only from context, say so if you can't" grounding instruction used in the RAG project.

## Results

| Question                                              | Extractive                                                  | Generative                                                     | Verdict                                                                     |
| ----------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Professional plan price per guard                     | `149` (conf. 0.75)                                        | "...costs 149 rupees per guard per month."                     | Both correct                                                                |
| How proxy attendance is prevented (spans 2 sentences) | "ML-powered face detection" (conf. 0.65)                    | Full synthesis of selfie requirement + detection mechanism     | Both correct, generative more complete                                      |
| Enterprise plan contents                              | **Starter plan's feature list** (conf. 0.43) — wrong | Correct, including "everything in Professional" inheritance    | **Extractive wrong**                                                  |
| Free trial credit card requirement (yes/no phrasing)  | Empty answer (conf. 0.92)                                   | "No... 'no credit card required'"                              | **Extractive failed, with high confidence**                           |
| Refund policy (not in the source at all)              | Empty answer (conf. 0.71)                                   | Explicitly states the context doesn't contain this information | Generative correctly honest; extractive gave no clear "I don't know" signal |

## The headline finding: extractive QA's confidence score is not a reliable signal of correctness

This is the most important and most counterintuitive result from this comparison. On the Enterprise-plan question, the model returned a **wrong** answer (the Starter plan's list, not Enterprise's) with moderate confidence (0.43). On the yes/no-phrased question, it returned an **empty, unusable** answer with *high* confidence (0.92) — the worst possible combination, since a high score gives no warning that something went wrong. A system that is simultaneously wrong and confident is more dangerous in practice than one that is wrong and clearly uncertain, because there's no signal telling a downstream user (or system) to double-check the answer.

## Where each approach actually succeeded or failed, and why

- **Extractive QA succeeded** when the question's phrasing closely mirrored the source text's phrasing, and the answer was a single contiguous span (the price question, and — partially — the proxy-attendance question).
- **Extractive QA failed** when the question was phrased differently from the source (yes/no framing over a declarative source sentence), when the true answer required selecting the *correct* one of several similarly-shaped list spans in the text (Enterprise vs. Starter), and when no answer existed in the text at all — in the last two cases, it produced either a wrong span or an empty one, without a dependably low confidence score to flag the failure.
- **Generative QA succeeded on every question tested**, including correctly identifying the one genuinely unanswerable question. Its core advantage here wasn't fluency — it was the ability to synthesize across sentences and correctly select among similarly-structured information (Starter's list vs. Enterprise's list), which extractive QA's single-span mechanism cannot do by design.
- **Generative QA's real risk**, not exercised by this particular test set but worth stating honestly: nothing in this comparison ruled out hallucination on a question where the LLM might generate a plausible-sounding but ungrounded detail. The "say so if you don't know" instruction worked correctly here, but that's a prompting choice, not a structural guarantee the way extractive QA's span-only mechanism is.

## What I'd improve with more time

- Test extractive QA's confidence score systematically across a larger set of answerable and unanswerable questions, to determine whether a usable confidence threshold exists for flagging likely failures, rather than relying on the two data points observed here.
- Try a larger or differently-trained extractive model to see whether the Starter-vs-Enterprise list confusion is specific to this smaller, distilled model.
- Test a question deliberately designed to tempt the generative model into hallucinating a specific number or detail not in the source, to directly probe the risk noted above rather than only its current strengths.

## Stack

Python, Hugging Face Transformers (DistilBERT fine-tuned on SQuAD), Groq API (Llama 3.3 70B), PyTorch
