import streamlit as st
import requests

st.title("Simple Chatbot")

user_input = st.text_input("Enter your message:")
if st.button("Send"):
    response = requests.post("http://127.0.0.1:8000/chat", json={"message": user_input})
    bot_reply = response.json().get("reply", "No response")
    st.write("🤖:", bot_reply)
