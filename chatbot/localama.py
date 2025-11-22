from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise, helpful assistant."),
    ("user", "Question: {question}")
])


st.title("LangChain + Ollama (light model)")
input_text = st.text_input("Ask something:")


llm = ChatOllama(
    model="gemma3:1b",     
    temperature=0.1,

    model_kwargs={
        "num_ctx": 1024,   
        "num_predict": 256  
    },
)

parser = StrOutputParser()
chain = prompt | llm | parser

if input_text:
    with st.spinner("Thinking..."):
        st.write(chain.invoke({"question": input_text}))