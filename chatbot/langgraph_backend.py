from langgraph.graph import StateGraph,START,END
from typing import  TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver#memory in ram 
from dotenv import load_dotenv
import sys
import os
from langchain_groq import ChatGroq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from get_llm import get_llm_answer
load_dotenv()#to load env variable from .env file
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]#all Message type like AImessage,HumanMessage,SystemMessage,ToolMessage inherit from BaseMessage,
    #aur hame reducer function add krna padega q ki ye purani chz hata deta and nayi chat arakh leta hai .
# llm=ChatOpenAI()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
# config={'configurable':{'thread_id':'thread_id-1'}}
def chat_node(state:ChatState):
    #take user query from state
    messages=state['messages']
    #send it to llm 
    # response=get_llm_answer("llama", messages[-1].content)
    response=llm.invoke(messages)
    #store response in state
    return {'messages':[response]}

checkpointer=MemorySaver()
graph=StateGraph(ChatState)
#add nodes
graph.add_node('chat_node',chat_node)

#add edge
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)
