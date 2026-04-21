from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 引入两个模型的配置
from config import get_cloud_llm, get_local_llm
from logger import get_logger
from state import RoadDiseaseState
from tools import get_retrieve_docs, predict_image_crack

log = get_logger(__name__)
tool_list = [predict_image_crack, get_retrieve_docs]
memory = MemorySaver()

# --- 新增：定义计划的数据结构 ---
class Plan(BaseModel):
    steps: List[str] = Field(description="解决当前道路病害问题的具体步骤列表，通常包含识别、查规范、给方案等。")

# --- 新增：规划节点 ---
def planner_node(state: RoadDiseaseState):
    log.info("进入 Planner 节点进行任务拆解")
    messages = state.get("messages", [])
    if not messages:
        return {"plan": []}
    
    user_input = messages[-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            你是一个公路养护规划专家。
            请根据用户的需求，将其拆解为2到3个具体的执行步骤。
            如果涉及图片必须先进行视觉识别，如果涉及评估必须查阅规范。
            视觉识别能够自动读取本地或者在线图片
          """),
        ("user", "{input}")
    ])
    
    # 使用本地小模型进行规划，并要求输出结构化数据
    planner = prompt | get_local_llm().with_structured_output(Plan)
    result = planner.invoke({"input": user_input})
    
    log.info(f"生成计划: {result.steps}")
    return {"plan": result.steps}

def reporter_node(state: RoadDiseaseState):
    log.info("进入 Reporter 节点，生成最终养护方案")
    messages = state.get("messages", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位资深的公路养护总工。
请仔细阅读对话历史中工具返回的 JSON 结构化数据（包含视觉识别结果和规范检索结果）。
请严格基于这些客观数据，输出一份专业的《路面病害评估与养护方案》。

报告要求结构清晰，包含：
1. 病害基本信息 (汇总识别到的类型、尺寸)
2. 规范判定依据 (引用检索到的规范条文)
3. 处治措施建议 (基于规范给出的具体施工步骤)

注意：
- 绝不允许编造任何尺寸或数据！如果工具数据为空或失败，请如实说明。
- 不要暴露底层的 JSON 格式，直接输出排版良好的专业文字报告。
- 不允许再调用任何工具。"""),
        MessagesPlaceholder("messages"),
    ])
    
    # 确保这里使用的是云端大模型，且没有 bind_tools
    chain = prompt | get_cloud_llm() 
    result = chain.invoke({"messages": messages})
    
    return {"maintenance_plan": result.content, "messages": [result]}

def create_agent(llm, tools):
    # 把系统提示词硬编码进去了，因为后续会动态注入状态
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
你是一个道路养护执行Agent。你需要严格按照提供的计划步骤，使用合适的工具来推进问题的回答。
在你的回答前加上FINAL ANSER，以便知道停止。
不要告诉用户你使用的是什么模型。
"""),
        ("system", "当前的执行计划是：\n{plan}"),
        MessagesPlaceholder("messages"),
    ])
    return prompt | llm.bind_tools(tools)

# 执行节点继续使用云端大模型
cloud_llm = get_cloud_llm()
agent = create_agent(cloud_llm, tool_list)

def agent_decision(state: RoadDiseaseState):
    messages = state["messages"]
    current_plan = "\n".join(state.get("plan", []))
    log.info("agent 节点执行，依据计划执行任务")
    
    # 将当前的计划注入到提示词中
    result = agent.invoke({"messages": messages, "plan": current_plan})
    return {"messages": result}

# --- 其他不变 ---
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
    
    # 注册节点
    builder.add_node("planner", planner_node)
    builder.add_node("agent", agent_decision)
    builder.add_node("reporter", reporter_node)
    builder.add_node("tools", tools_node)
    
    
    # 编排流程
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", "end": "reporter"})
    builder.add_edge("reporter", END)
    
    graph = builder.compile(checkpointer=memory)
    if need_draw:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_bytes)
        log.info("已生成流程图: workflow.png")
    return graph

if __name__ == "__main__":
  build_graph(True)