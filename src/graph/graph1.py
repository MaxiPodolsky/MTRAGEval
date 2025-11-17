import uuid
from typing import Literal, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from graph.agent_node import agent_node
from graph.generate_node import generate
from graph.get_human_message import get_last_human_message
from graph.graph_state import AgentState, Grade
from graph.rewrite_node import rewrite
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log
from utils.print_utils import _print_event


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determine whether the retrieved documents are relevant to the question.
    """
    log.info("---Check document relevance---")
    
    # prompt = PromptTemplate(
    #     template="""你是一个评估检索文档与用户问题相关性的评分器。

    #     文档内容：{context}
    #     用户问题：{question}
    #     请根据以下标准判断文档是否与问题相关：
    #     - 如果文档包含与用户问题相关的关键词或语义含义，评为相关
    #     - 如果文档内容与问题无关或信息不足，评为不相关

    #     请严格按照Grade类的格式要求，只输出"yes"或"no"来表示相关性评分。

    #     你的判断：""",
    #     input_variables=["context", "question"],
    # )

    prompt = PromptTemplate(
        template="""You are a scorer that assesses the relevance of retrieved documents to user questions.

        Document content:{context}
        User issue:{question}
        Please determine the relevance of a document to the question based on the following criteria:
        - If the document contains keywords or semantic meanings relevant to the user's question, it is rated as relevant.
        - If the document content is irrelevant to the question or lacks sufficient information, it is rated as irrelevant.
        
        Please strictly adhere to the formatting requirements for the Grade class and only output "yes" or "no" to indicate the relevance score.

        Your judgment:""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm

    messages = state["messages"]
    last_message = messages[-1]

    question = get_last_human_message(messages).content
    docs = last_message.content

    response = chain.invoke({"question": question, "context": docs})
    score_text = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
    
    # Clean up and validate the response
    score_text = score_text.replace('"', '').replace("'", "").strip()
    
    # Ensure the output conforms to the constraints of the Grade class.
    if score_text not in ["yes", "no"]:
        print(f"---Warning: The model returned an unexpected response.: '{score_text}'，use standardization processing---")
        # 更精确的启发式判断
        positive_indicators = ["yes", "是", "相关", "relevant", "true", "正确", "匹配"]
        negative_indicators = ["no", "否", "不相关", "irrelevant", "false", "错误", "不匹配"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in score_text)
        negative_count = sum(1 for indicator in negative_indicators if indicator in score_text)
        
        if positive_count > negative_count:
            score_text = "yes"
        else:
            score_text = "no"
    
    # Manually create Grade objects to maintain compatibility with existing code.
    scored_result = Grade(binary_score=score_text)
    
    print(f"---Relevance score: {scored_result.binary_score}---")
    
    if scored_result.binary_score == "yes":
        print("---Output: Document related---")
        return "generate"
    else:
        print("---Output: Document irrelevant---")
        return "rewrite"



# Define a new workflow diagram
workflow = StateGraph(AgentState)

# Add node
workflow.add_node('agent', agent_node)
workflow.add_node('retrieve', ToolNode([retriever_tool]))
workflow.add_node('rewrite', rewrite)
workflow.add_node('generate', generate)

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools': 'retrieve',
        END: END
    }
)

workflow.add_conditional_edges(
    'retrieve',
    grade_documents,
)

workflow.add_edge('rewrite', 'agent')
workflow.add_edge('generate', END)



memory = MemorySaver()

# Compilation state diagram, checkpoint configured as memory, breakpoint configured.
graph = workflow.compile(checkpointer=memory)


# draw_graph(graph, 'graph_rag1-2.png')
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}

_printed = set()  # use set, avoid duplicate printing

# Execute workflow
while True:
    question = input('User :')
    if question.lower() in ['q', 'exit', 'quit']:
        log.info('End of conversation, bye!')
        break
    else:
        inputs = {
            "messages": [
                ("user", question),
            ]
        }
        events = graph.stream(inputs, config=config, stream_mode='values')
        # print information
        for event in events:
            _print_event(event, _printed)