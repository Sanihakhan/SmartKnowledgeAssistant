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

# Set Streamlit page config
st.set_page_config(page_title="Smart Assistant", page_icon="🎀", layout="centered")

# Theme toggle
theme = st.sidebar.radio("🎨 Choose Theme", ["Light Mode", "Dark Mode"])

if theme == "Light Mode":
    st.markdown("""
        <style>
            .stApp { background-color: #fff0f5; color: #000; }
            .title-text { color: #FF69B4 !important; text-align: center; }
            .subtitle-text { color: #222 !important; text-align: center; }
            .user-input-box input { background-color: #fff0f5; color: black; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .title-text { color: #FF69B4 !important; text-align: center; }
            .subtitle-text { color: white !important; text-align: center; }
        </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title-text'>🎀 Smart Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Ask about company policies or employee stats 📒</p>", unsafe_allow_html=True)
st.divider()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load FAISS RAG index
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

# Input box inside form (safe clearing)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("📝 You:", placeholder="Ask anything...", key="input_text")
    submit = st.form_submit_button("Ask")

if submit and user_input:
    query = user_input.lower()

    # CSV logic
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

    # Add latest on top
    st.session_state.chat_history.insert(0, ("Bot", response))
    st.session_state.chat_history.insert(0, ("You", user_input))

# Show messages (latest on top)
for sender, msg in st.session_state.chat_history:
    icon = "🎀" if sender == "Bot" else "📝"
    with st.chat_message(icon):
        st.markdown(f"**{sender}:** {msg}")





#.venv\Scripts\activate
#streamlit run app/app.py