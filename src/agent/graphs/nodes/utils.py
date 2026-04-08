# Graph utility functions

import json
from json.decoder import JSONDecodeError

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

def get_recent_messages(messages: list[BaseMessage], max_turns: int = 50) -> list[BaseMessage]:
    """This function returns the recent messages sent by the user"""
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    sliced = non_system[-(max_turns * 2):]

    # Walk forward until the first HumanMessage — drops any orphaned
    # ToolMessages or AIMessages with tool_calls at the slice boundary
    for i, msg in enumerate(sliced):
        if isinstance(msg, HumanMessage):
            return sliced[i:]

    return sliced

def trim_tool_message(messages: list[BaseMessage]) -> list[BaseMessage]:
    trimmed = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
                if "result" in payload and payload["result"]:
                    payload["result"]["value"] = "[truncated]"
                payload.pop("stdout", None)
                trimmed_content = json.dumps(payload)
            except (JSONDecodeError, TypeError):
                trimmed_content = msg.content
            trimmed.append(ToolMessage(content=trimmed_content, tool_call_id=msg.tool_call_id))
        else:
            trimmed.append(msg)
    return trimmed