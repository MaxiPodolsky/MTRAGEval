from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an intelligent assistant. Use the tools to answer user questions as best as you can.'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),  #scratchpad is for intermediate steps
])

agent = create_tool_calling_agent(llm, [retriever_tool], prompt)

executor = AgentExecutor(agent=agent, tools=[retriever_tool])


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_history = RunnableWithMessageHistory(  
    executor,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)


#test
resp = agent_with_history.invoke(  
    {'input': 'What happened after the French Revolution?'},
    config={'configurable': {"session_id": '11111'}}
)

print(resp)