from pypdf import PdfReader
import os

papers_folder = "papers"
output_folder = "extracted_text"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(papers_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(papers_folder, filename)
        reader = PdfReader(path)
        
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        output_name = filename.replace(".pdf", ".txt")
        output_path = os.path.join(output_folder, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        print(f"Extracted {filename} -> {output_name} ({len(full_text)} characters)")