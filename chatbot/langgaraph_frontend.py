import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import BaseMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()#to load env variable from .env file

CONFIG={'configurable':{'thread_id':f'thread_id-1'}}
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
# message_history=[]
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input=st.chat_input('Type your message here...')
if user_input:
    #first add the message to message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
        
    # config={'configurable':{'thread_id':thread_id}}
    response=chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG)
    ai_message=response['messages'][-1].content
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
        