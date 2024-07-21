import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_vectorstore_from_pdf(pdf_file):
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name  # Get the path of the temporary file

        # Load documents using the file-like object
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split the documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = Chroma.from_documents(
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./data"
        )
        vector_store.persist()
        os.remove(temp_file_path)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "You are an exceptional business support chatbot for a wine business. Answer questions using only the information from the provided PDF. For questions outside the scope of the PDF, politely direct the user to contact the business directly.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context below:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    # Create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']

# App Config 
st.set_page_config(page_title="Wine Business Chatbot")
st.title("Wine Business Chatbot")

# Sidebar 
with st.sidebar:  
    st.header("Settings")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file is None:
        st.info("Please upload a PDF file")
    else:
        if "pdf_name" in st.session_state and st.session_state.pdf_name != pdf_file.name:
            st.session_state.pop("vector_store", None)
            st.session_state.pop("chat_history", None)
        
        if st.button("Preprocess"):
            st.session_state.vector_store = get_vectorstore_from_pdf(pdf_file)
            if st.session_state.vector_store:
                st.session_state.pdf_name = pdf_file.name
                st.success("PDF processed successfully!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="Hello, I am a bot. How can I help you?")]

if st.session_state.get("vector_store") is None:
    st.info("Please preprocess the PDF by clicking the 'Preprocess' button in the sidebar.")
else:
    user_query = st.text_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(SystemMessage(content=response))

    # Conversation 
    for message in st.session_state.chat_history:
        if isinstance(message, SystemMessage):
            with st.chat_message("system"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)
