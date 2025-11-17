from graph.graph_state import AgentState
from langchain_core.messages import HumanMessage
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log


# agent节点
def agent_node(state: AgentState):    #轻量级不是很复杂的，可以用函数的写法
    """
    The intelligent agent model is invoked to generate a response based on the current state. Depending on the question,
    it will decide whether to use a retrieval tool or terminate the process directly.

    Parameters:
        state (messages): current state

    Returns:
        dict: updated state containing the agent's response appended to messages
    """
    log.info("---Enter the workflow---")
    messages = state.get("messages", [])

    # If `force_generate` is set in the state, the retrieval tool is skipped, and `final_query` is used directly to call the LLM to generate the LLM.
    if state.get("force_generate"):
        final_q = state.get("final_query") or (messages[-1].content if messages else "")
        log.info("force_generate is present ，skip the search tool and generate the answer directly.")
        msg = HumanMessage(content=final_q)
        response = llm.invoke([msg])
        # return and clear the force_generate flag
        return {"messages": [response], "force_generate": False, "final_query": None}

    # default behavior: bind the retrieval tool and let the model decide whether to call it
    model = llm.bind_tools([retriever_tool])
    #response = model.invoke([messages[-1]])
    response = model.invoke(messages)  #load all messages
    return {"messages": [response]}