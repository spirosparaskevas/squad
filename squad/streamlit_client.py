import streamlit as st
import requests

st.title("Question Answering App")
with st.form(key="request_form"):
    context = st.text_area("Context", height=200)
    question = st.text_input("Question")
    submit_button = st.form_submit_button(label="Ask")

if submit_button:
    r = requests.post(f"http://127.0.0.1:8080/squad/predict", json={"context": context, "question": question})
    st.write(r.json())
