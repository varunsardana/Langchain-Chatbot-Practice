

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()  

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server",
)

 
model = ChatOllama(
    model="gemma3:1b",
    temperature=0.1,

    num_ctx=1024,
    num_predict=256,
)

# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise, helpful assistant."),
    ("user", "Question: {question}")
])


chain = prompt | model


add_routes(app, chain, path="/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)




