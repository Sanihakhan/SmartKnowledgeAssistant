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

def main():
    # Load embeddings and vector store (for policies.pdf)
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    # Memory for chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # RAG-based chatbot
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    # Load structured employee data from CSV
    employee_df = pd.read_csv("data/Employee.csv")

    print("💬 Ask your internal query (type 'exit' to quit):")
    while True:
        query = input("🧑‍💼 You: ")
        if query.lower() == "exit":
            break

        # ✅ Handle CSV-based logic
        q_lower = query.lower()

        if "leave rate" in q_lower:
            total = len(employee_df)
            left = employee_df["LeaveOrNot"].sum()
            rate = (left / total) * 100
            print(f"🤖 Bot: The employee leave rate is approximately {rate:.2f}%.")
            continue

        if "how many employees" in q_lower and "left" in q_lower:
            count = employee_df["LeaveOrNot"].sum()
            print(f"🤖 Bot: {int(count)} employees have left the company.")
            continue

        if "average experience" in q_lower:
            avg = employee_df["ExperienceInCurrentDomain"].mean()
            print(f"🤖 Bot: The average experience in current domain is {avg:.2f} years.")
            continue

        if "female" in q_lower and "left" in q_lower:
            filtered = employee_df[(employee_df["Gender"] == "Female") & (employee_df["LeaveOrNot"] == 1)]
            print(f"🤖 Bot: {len(filtered)} female employees have left.")
            continue

        if "total employees" in q_lower:
            print(f"🤖 Bot: There are {len(employee_df)} total employees in the dataset.")
            continue

        # ✅ Otherwise, use RAG to answer from PDF
        response = qa.invoke({"question": query})
        print("🤖 Bot:", response["answer"])

if __name__ == "__main__":
    main()


#What is the employee leave rate?

#How many employees have left?

#How many female employees left?

#What is the average experience?

#How many total employees are there?

#What topics or rules are discussed in the policies?

#Summarize the document.

#What does it say about document management?

