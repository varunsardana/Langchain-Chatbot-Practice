import streamlit as st
import os #lets you read environment variables
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv #to load environment variables from .env file
load_dotenv()

## load the groq api key from environment variable

groq_api_key = os.getenv('GROQ_API_KEY') # we load the groq api key from environment variable
if not groq_api_key:
    st.error("Missing GROQ_API_KEY in environment. Add it to your .env file")
    st.stop()



if "vector" not in st.session_state: # we use session state to store the vector store so that we don't have to reload it every time the user interacts with the app
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # for faiss we give the retrived final documents and the embeddings to store the documents in the vector store

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key= groq_api_key, model="gemma2-9b-it")

qa_prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the context.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, qa_prompt, document_variable_name="context",)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain) # so you have the llm, and document chain (which has the prompt which furhter includes the retrieved chunks along with the users quesiton)


user_question = st.text_input("Enter your question here")

response = None
if user_question:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_question})
    st.write(response["answer"])
    st.caption(f"Response time: {time.process_time() - start:.2f}s")

# With streamlit expander: show only if we have a response
if response is not None:
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------------")






