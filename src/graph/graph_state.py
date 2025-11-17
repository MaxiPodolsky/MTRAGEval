from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):   #Represents the state of the agent's workflow
    #  The `add_messages` function defines how state updates should be handled.
    # `add_messages` means "append".

    messages: Annotated[list[BaseMessage], add_messages]  
            #BaseMessage is a class in langchain_core.messages. It is the base class for all messages.


class Grade(BaseModel):
    """Binary scoring of correlation test"""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")