#from langchain_community.tools import TavilySearchResults
import os

from langchain_openai import ChatOpenAI

from utils.env_utils import OPENAI_API_KEY, DEEPSEEK_API_KEY

# llm = ChatOpenAI(  # openai的
#     temperature=0,
#     model='gpt-4o-mini',
#     api_key=OPENAI_API_KEY,
#     base_url="https://xiaoai.plus/v1")


#web_search_tool = TavilySearchResults(max_results=2)

# llm = ChatOpenAI(
#    temperature=0.5,
#    model='deepseek-chat',
#    api_key=DEEPSEEK_API_KEY,
#    base_url="https://api.deepseek.com")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
)
