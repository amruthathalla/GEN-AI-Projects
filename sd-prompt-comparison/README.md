
# Stable Diffusion Prompt Engineering: A Controlled Comparison

A systematic test of how specific prompt-engineering techniques — negative prompts, style modifier keywords, and their combination — actually change Stable Diffusion's output, holding the subject constant across all variants. Built as a controlled comparison rather than a one-off "type a prompt, get an image" demo.

## Why this project

Text-to-image generation is easy to demo and hard to actually understand without testing it systematically. Rather than generate one image and call it done, this project holds a single subject constant and varies one technique at a time, so any visible difference between outputs can be attributed to a specific, named cause — the same experimental discipline used elsewhere in this portfolio (chunk-size testing in the RAG project, precision/recall analysis in the spam classifier) applied to image generation.

## Setup

- **API**: Stability AI's Stable Image Core endpoint (hosted, no local GPU required)
- **Subject held constant across all variants**: "a portrait of an elderly fisherman"
- **Variants tested**:
  1. Baseline — plain prompt, no technique applied
  2. + Negative prompt (`blurry, deformed face, extra fingers, cartoon, low quality, distorted`)
  3. + Style modifier keywords (`cinematic lighting, photorealistic, highly detailed, 8k, professional photography`)
  4. Combined — style modifiers and negative prompt together

## Results

![Comparison grid](outputs/comparison_grid.png)

**Negative prompting produced a clear, visible improvement.** The version with an explicit negative prompt was noticeably cleaner and more realistic than the baseline, with fewer visible artifacts — consistent with negative prompts working precisely because they target known, common failure modes (facial distortion, anatomical errors) that diffusion models are prone to, rather than just adding more descriptive text.

**Style modifier keywords had little visible effect on this specific endpoint.** Adding common style/quality keywords ("cinematic lighting," "8k," "professional photography") produced an image that was hard to distinguish from the baseline. The likely explanation: Stable Image Core appears to default to a fairly polished, photographic look already, leaving less headroom for these particular keywords to add a visible difference — a ceiling effect, not a failure of the technique itself. This is a genuinely useful finding precisely because it's not the result one might assume going in.

**The combined version mainly reflected the negative prompt's effect**, with no clear additional gain from also including style modifiers — consistent with the style-modifier result on its own.

## A real limitation worth naming: no CFG scale exposed

Stability's Stable Image Core endpoint (used throughout this comparison) does not expose a CFG scale parameter — a setting that controls how strictly the model follows the prompt versus exploring its own variation, available on Stability's older SDXL/SD1.6 endpoints but not on this simplified, more automated one. This was discovered while designing the experiment, not assumed beforehand, and is reported here rather than silently dropped: it's a genuine constraint of the specific endpoint chosen, not a gap in the testing methodology.

## What I'd improve with more time

- Repeat this comparison using Stability's older `/v1/generation` endpoints specifically to test CFG scale as a fifth variable, now that Core's limitation is known.
- Test the same technique set on a second, different subject (e.g., a landscape or object) to check whether the "style modifiers barely matter" finding holds generally, or is specific to portraits.
- Run each variant multiple times (not once) to check how much output varies between identical calls, since diffusion models are stochastic by default — a single image per variant can't distinguish a real effect from random variation.

## Stack

Python, Stability AI API (Stable Image Core), Pillow (for building the comparison grid), python-dotenv
