import time
import streamlit as st
from utils import load_chain

company_logo = 'https://influencermarketinghub.com/agency/wp-content/uploads/2023/03/102376.jpg'

st.set_page_config(
    page_title="Xuberan Notion Chatbot",
    page_icon=company_logo
)

params = st.experimental_get_query_params()

try:
    partner = dict(params)["partner"][0]
except (KeyError, TypeError):
    partner = ""

chain = load_chain()

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": f"Hi {partner}! I am Xuberan's smart AI. How can I help you today?"}]

for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if query := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        result = chain({"question": query})
        response = result['answer']
        full_response = ""

        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": response})