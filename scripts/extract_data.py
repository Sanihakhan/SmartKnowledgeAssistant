import fitz  # PyMuPDF
import pandas as pd
import os

def extract_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

def extract_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def extract_all():
    texts = []
    for filename in os.listdir("data"):
        path = os.path.join("data", filename)
        if filename.endswith(".pdf"):
            texts.append(extract_from_pdf(path))
        elif filename.endswith(".csv"):
            texts.append(extract_from_csv(path))
    return texts

if __name__ == "__main__":
    all_texts = extract_all()
    for i, txt in enumerate(all_texts):
        with open(f"data/clean_text_{i}.txt", "w", encoding="utf-8") as f:
            f.write(txt)


#data/clean_text_0.txt
#data/clean_text_1.txt