from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from graph.get_human_message import get_last_human_message
from llm_models.all_llm import llm
from utils.log_utils import log
#from langchain_community.memory import ConversationBufferWindowMemory
#from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def generate(state):
    """
    Generate the final answer based on the retrieved documents and the user's question.

    Parameters:
        state (messages): current state

    Returns:
         dict: Updated state containing restatement of the problem
    """
    log.info("---Generate the final answer---")
    messages = state["messages"]
    question = get_last_human_message(messages).content
    last_message = messages[-1]

    docs = last_message.content

    
    prompt = PromptTemplate(
        template="You are an multi-turn assistant for question answering tasks. Please answer the question based on the following retrieved context. If you don't know the answer, please state so directly. Keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:",
        input_variables=["question", "context"], 
    )
    
    
    # chain processing
    rag_chain = prompt | llm | StrOutputParser()
    # implementation
    response = rag_chain.invoke({"context": docs, "question": question})
    ai_message = AIMessage(content=response)
    return {"messages": [ai_message]}