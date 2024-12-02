import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory # use for chat history
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader  

from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ-KEY")
os.environ['HF_API_KEY'] = os.getenv('HF-TOKEN')

groq_api_key = os.getenv("GROQ-KEY")
hf_api_key = os.getenv("HF-TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")


#llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

st.title("Conversation RAG woth pdf")
st.write("Upload pdf")

api_key = st.text_input("Enter your Groq API key")

if api_key:
    llm = ChatGroq(groq_api_key = groq_api_key , model_name="Gemma-7b-It")
    session_id = st.text_input("Session ID ", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store ={}

    uploaded_files = st.file_uploader("choose a PDF file", type = "pdf" , accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open (temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectors = FAISS.from_documents(document=splits, embeddings= embeddings)
        retriever=vectors.as_retriever()

    contextualize_q_system_prompt=(
        "Given a chat history and the latest  user question"
        "Which might refrence the context in the chat history"
        "formulate a standalone question which can be understood"
        "without the chain history donot answer the question"
        "just formulate it if needed otherwise return it as it is"

    )

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system" , contextualize_q_system_prompt),
             MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]

    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


    system_prompt= (
        "You are an assisstant for question answer tasks"
        "Use the following piece of retrieved context to answer"
        "the question. if you donot know the answer , say that you"
        "donot know. use three senetnces maximum and keep the answer"
        "concise"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chathistory",
        output_messages_key="answer"

    )

    user_input = st.text_input("your question")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )

        st.write(st.session_state.store)
        st.success("Assisstant:", response['answer'])
        st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter your Groq API key")



