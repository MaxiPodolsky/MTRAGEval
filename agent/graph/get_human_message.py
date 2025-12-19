from typing import List

from langchain_core.messages import BaseMessage, HumanMessage


def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    # reverse iterate to find the last HumanMessage
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    raise ValueError("No HumanMessage found in the messages list")