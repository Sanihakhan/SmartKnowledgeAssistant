from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def load_texts():
    texts = []
    for filename in os.listdir("data"):
        if filename.startswith("clean_text_"):
            with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def main():
    texts = load_texts()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks += splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("vector_store")

if __name__ == "__main__":
    main()


