from langchain_core.messages import HumanMessage

import os

from graph.get_human_message import get_last_human_message
from llm_models.all_llm import llm
from utils.log_utils import log



def rewrite(state):
    """
        Transform the query to generate a better question.
        Parameters:
            state (messages): Current state
        Returns:
            dict: An updated state containing restated questions
    """

    log.info("---rewrite query---")
    messages = state["messages"]
    question = get_last_human_message(messages).content

    #原来这一段是没有的
    # Limit the number of rewrites (configurable via the environment variable REWRITE_MAX_ATTEMPTS, default is 3).
    try:
        MAX_REWRITES = int(os.getenv("REWRITE_MAX_ATTEMPTS", "3"))
    except Exception:
        MAX_REWRITES = 3

    # If this is a new user question (different from the last_user_query recorded in the state),
    #then reset rewrite_count to 0 so that each new question will be counted separately.
    #prev_query = state.get("last_user_query")
    prev_query = get_last_human_message(messages).content
    if prev_query != question:
        rewrite_count = 0
    else:
        # Otherwise, continue using the existing count (or 0 if no count exists).
        rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= MAX_REWRITES:
        log.info(f"The maximum number of rewrites has been reached.({rewrite_count})， will proceed directly to the generation phase, using the original problem.")
        # Returning to the original question, a forced generation flag is set, instructing the agent to skip the retrieval and generate directly.
        return {
            "messages": [HumanMessage(content=question)],
            "rewrite_count": rewrite_count,
            "last_user_query": question,
            "force_generate": True,
            "final_query": question,
        }

    msg = [
        HumanMessage(
            content=f""" \n 
    Analyze the input and attempt to understand the underlying semantic intent/meaning. 。\n 
    This is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Please suggest an improved version of the problem. (you can focus on past human message to understand what the user want to ask .): """,
        )
    ] 


    # Scoring model (rewritten using LLM)
    response = llm.invoke(msg)
    # Increment the rewrite count and return, while also recording last_user_query.
    return {"messages": [response], "rewrite_count": rewrite_count + 1, "last_user_query": question}
