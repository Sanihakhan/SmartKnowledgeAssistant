import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import os

# Load environment variables
load_dotenv()
employee_df = pd.read_csv("data/Employee.csv")

# Sidebar for theme toggle
st.set_page_config(page_title="Smart Assistant", page_icon="🎀", layout="centered")
theme = st.sidebar.radio("🌓 Choose Theme", ["Light Mode", "Dark Mode"])

# Inject light pink CSS if in Light Mode
if theme == "Light Mode":
    st.markdown(
        """
        <style>
        body {
            background-color: #fff0f5;
        }
        .stApp {
            background-color: #fff0f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Title and description
st.markdown(
    "<h1 style='text-align: center; color: #FF69B4;'>🎀 Smart Knowledge Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Ask about company policies or employee stats 📒</p>",
    unsafe_allow_html=True,
)
st.divider()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load vector store (for RAG)
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
)

# Input box
user_input = st.text_input("📝 You:", placeholder="Ask anything...")

if user_input:
    query = user_input.lower()

    # CSV-based logic
    if "leave rate" in query:
        total = len(employee_df)
        left = employee_df["LeaveOrNot"].sum()
        rate = (left / total) * 100
        response = f"Approximately {rate:.2f}% of employees have left the company."
    elif "how many employees" in query and "left" in query:
        count = employee_df["LeaveOrNot"].sum()
        response = f"{int(count)} employees have left the company."
    elif "average experience" in query:
        avg = employee_df["ExperienceInCurrentDomain"].mean()
        response = f"The average experience in the current domain is {avg:.2f} years."
    elif "female" in query and "left" in query:
        filtered = employee_df[(employee_df["Gender"] == "Female") & (employee_df["LeaveOrNot"] == 1)]
        response = f"{len(filtered)} female employees have left."
    elif "total employees" in query:
        response = f"There are {len(employee_df)} total employees in the dataset."
    else:
        result = qa_chain.invoke({"question": user_input})
        response = result["answer"]

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display messages
for sender, msg in st.session_state.chat_history:
    icon = "🎀" if sender == "Bot" else "📝"
    with st.chat_message(icon):
        st.markdown(f"**{sender}:** {msg}")
