from PIL import Image, ImageDraw, ImageFont
import os

image_files = [
    ("outputs/01_baseline.png", "1. Baseline"),
    ("outputs/02_negative_prompt.png", "2. + Negative Prompt"),
    ("outputs/03_style_modifiers.png", "3. + Style Modifiers"),
    ("outputs/04_combined.png", "4. Combined"),
]

# Load all images and standardize their size
thumb_size = (400, 400)
images = []
for path, label in image_files:
    img = Image.open(path).convert("RGB")
    img = img.resize(thumb_size)
    images.append((img, label))

# Layout: 2x2 grid, with space below each image for its label
label_height = 40
padding = 10
grid_width = thumb_size[0] * 2 + padding * 3
grid_height = (thumb_size[1] + label_height) * 2 + padding * 3

grid = Image.new("RGB", (grid_width, grid_height), color="white")
draw = ImageDraw.Draw(grid)

try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # col, row

for (img, label), (col, row) in zip(images, positions):
    x = padding + col * (thumb_size[0] + padding)
    y = padding + row * (thumb_size[1] + label_height + padding)
    grid.paste(img, (x, y))
    text_x = x + thumb_size[0] // 2
    text_y = y + thumb_size[1] + 8
    draw.text((text_x, text_y), label, fill="black", font=font, anchor="mm")

os.makedirs("outputs", exist_ok=True)
grid.save("outputs/comparison_grid.png")
print("Saved: outputs/comparison_grid.png")