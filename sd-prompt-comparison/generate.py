import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("STABILITY_API_KEY")
API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

def generate_image(prompt: str, negative_prompt: str = "", output_path: str = "output.png", aspect_ratio: str = "1:1"):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "image/*",
        },
        files={"none": ""},  # required by Stability's API even with no file upload
        data={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        },
    )

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {output_path}")
        return output_path
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    generate_image(
        prompt="a portrait of an girl with a red hat, in the style of van gogh",
        output_path="outputs/00_baseline.png",
    )