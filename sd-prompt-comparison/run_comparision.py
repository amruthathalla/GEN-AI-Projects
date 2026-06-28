from generate import generate_image
import os

os.makedirs("outputs", exist_ok=True)

base_prompt = "a portrait of an elderly fisherman"

# 1. Baseline -- already generated, but included here for a complete, reproducible script
generate_image(
    prompt=base_prompt,
    output_path="outputs/01_baseline.png",
)

# 2. Negative prompt -- same prompt, but now explicitly steering away from common SD problems
generate_image(
    prompt=base_prompt,
    negative_prompt="blurry, deformed face, extra fingers, cartoon, low quality, distorted",
    output_path="outputs/02_negative_prompt.png",
)

# 3. Style modifier keywords -- same subject, additional descriptive/quality language
generate_image(
    prompt=f"{base_prompt}, cinematic lighting, photorealistic, highly detailed, 8k, professional photography",
    output_path="outputs/03_style_modifiers.png",
)

# 4. Style modifiers + negative prompt combined
generate_image(
    prompt=f"{base_prompt}, cinematic lighting, photorealistic, highly detailed, 8k, professional photography",
    negative_prompt="blurry, deformed face, extra fingers, cartoon, low quality, distorted",
    output_path="outputs/04_combined.png",
)

print("\nAll variants generated. Check the outputs/ folder.")