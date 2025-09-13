pip install faiss-cpu




policies.pdf
     ↓ (PyMuPDF)
Clean Text
     ↓ (TextSplitter)
Chunks
     ↓ (OpenAI Embeddings)
Vectors
     ↓ (FAISS)
Vector DB
     ↓ (Query from user)
Similar Chunks
     ↓
OpenAI LLM ➝ Final Answer (Chat)



SmartKnowledgeAssistant/
├── data/ # Contains policies.pdf and other raw files
├── scripts/ # All logic scripts
│ ├── extract_data.py # Extracts text from PDFs
│ ├── vectorize.py # Chunks text and creates vector store
│ └── chat_bot.py # Conversational RAG bot
├── vector_store/ # FAISS vector database (auto-generated)
├── .env # Stores OpenAI API key
├── requirements.txt # Dependencies
└── README.md # Project documentation


policies.pdf ➝ PyMuPDF ➝ LangChain ➝ FAISS ➝ OpenAI LLM

📦 Tech Stack
LangChain (RAG logic): Breaks Text into Chunks Used in scripts/vectorize.py
FAISS (Semantic search)       
OpenAI GPT (LLM)
PyMuPDF (PDF text extraction)
dotenv (Environment management)



🚀 Future Additions
Integrate Employee.csv analysis
Add Streamlit UI


where is it storing conversational memory? (conversational memory is stored in RAM (runtime) using this line)
in chatbot code- 
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
ConversationBufferMemory	Stores previous messages in a Python list (temporarily)

This memory is then passed into:
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    memory=memory
)
LangChain automatically uses this memory to keep context of previous questions/answers.

in my project, conversational memory is stored in volatile memory (RAM) using LangChain’s ConversationBufferMemory during a single chat session.
if i restart the app, memory is wiped.

