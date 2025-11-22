import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat/invoke" 

def get_ollama_response(input_text: str) -> str:
    try:
        r = requests.post(API_URL, json={"input": {"question": input_text}}, timeout=60)
    except requests.exceptions.ConnectionError:
        return "❌ Could not reach the API server. Is it running on :8000?"


    try:
        data = r.json()
    except Exception:
        return f"❌ Non-JSON response ({r.status_code}): {r.text}"

    if r.status_code != 200:
        return f"❌ {r.status_code}: {data}"


    out = data.get("output")
    if isinstance(out, dict) and "content" in out:
        return out["content"]
    if isinstance(out, str):         
        return out

  
    return str(data)

st.title("LangChain + Ollama (API)")
input_text = st.text_input("Ask something:")

if input_text:
    st.write(get_ollama_response(input_text))