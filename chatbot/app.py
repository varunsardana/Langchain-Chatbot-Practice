from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
import os
from dotenv import load_dotenv


# Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the queries."),
        ("user", "Question:{question}"),

        ]
)

#sttreamlit framework

st.title("Chatbot with Langchain and OpenAI")
input_text = st.text_input("Search the topic you want")

#openAI LLM model
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain=prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))