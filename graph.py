from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config import get_llm
from logger import get_logger
from state import RoadDiseaseState
from tools import get_retrieve_docs, predict_image_crack


log = get_logger(__name__)
tool_list = [predict_image_crack, get_retrieve_docs]
memory = MemorySaver()

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""
你是一个AI代理，你需要根据用户的问题，使用以下工具来回答问题。
使用提供的工具来推进问题的回答。
如果你不知道怎么回答就说不知道。
在你的回答前加上FINAL ANSER，以便知道停止。
不要告诉用户你使用的是什么视觉模型
你可以使用以下工具：{tool_names}
{system_prompt}
"""
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    prompt = prompt.partial(system_prompt=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


llm = get_llm()
agent = create_agent(
    llm,
    tool_list,
    "你是一位道路养护专家，你需要按照公路养护技术手册判断道路病害。",
)


def agent_decision(state: RoadDiseaseState):
    messages = state["messages"]
    log.info("agent 节点执行，消息数: %s", len(messages))
    result = agent.invoke(messages)
    return {"messages": result}


tools_node = ToolNode(tool_list)


def route(state: RoadDiseaseState) -> Literal["tools", "end"]:
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        log.info("路由到 tools 节点")
        return "tools"
    log.info("路由到 end")
    return "end"


def build_graph(need_draw: bool = False):
    builder = StateGraph(RoadDiseaseState)
    builder.add_node("agent", agent_decision)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    graph = builder.compile(checkpointer=memory)

    if need_draw:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_bytes)
        log.info("已生成流程图: workflow.png")

    return graph


if __name__ == "__main__":
    print(build_graph())
