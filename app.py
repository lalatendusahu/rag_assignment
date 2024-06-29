import os
import sys
from urllib.request import urlretrieve
import streamlit as st
from dotenv import load_dotenv
from utils import load_pdf,text_chunk,vector_store,get_retrieval_qa


load_dotenv()
key_sec = os.getenv('SEC_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = key_sec


def main():
    """
    This is main functions to deploy streamlit application
    """
    st.header("AI Book Query chatbot💁")
    
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question from concept of Biology!"}
        ]
        
    if user_question := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant": # Stores chat history on the app
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_retrieval_qa().invoke({"query": user_question})
                st.write(result['result'])
                message = {"role": "assistant", "content": result['result']}
                st.session_state.messages.append(message)

    with st.sidebar:          #Display sidebar on app to take input on number of pages
        st.title("Menu:")
        num_page = st.text_input("Input number of pages till which you want to index and Click on the Submit & Process Button",key="num_page")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                selected_page = load_pdf(num_page)
                text_chunks = text_chunk(selected_page)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()