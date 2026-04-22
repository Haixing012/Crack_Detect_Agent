from typing import Literal, List
from functools import partial
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 引入配置和工具定义（仅引用函数/类，不触发初始化）
from config import get_cloud_llm, get_local_llm
from logger import get_logger
from state import RoadDiseaseState

log = get_logger(__name__)


# --- 定义计划的数据结构 ---
class Plan(BaseModel):
    steps: List[str] = Field(description="解决当前道路病害问题的具体步骤列表，通常包含识别、查规范、给方案等。")


def _get_tool_list():
    """惰性获取工具列表，避免模块导入时加载 YOLO 模型"""
    from tools import predict_image_crack, get_retrieve_docs
    return [predict_image_crack, get_retrieve_docs]


# --- 规划节点 ---
def planner_node(state: RoadDiseaseState):
    log.info("进入 Planner 节点进行任务拆解")
    messages = state.get("messages", [])
    if not messages:
        return {"plan": []}
    
    user_input = messages[-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            你是一个公路养护规划专家。
            现在需要你拆解用户的输入意图，并给出后续的执行步骤。
            如果涉及图片必须先进行视觉识别，如果涉及评估必须查阅规范。
            视觉识别能够自动读取本地或者在线图片
            你必须严格按照 **JSON 格式** 输出结果。
            不要输出任何解释、说明或多余文本。

            输出格式示例：
            {{
              "steps": ["步骤1xxxx", "步骤2xxxx", "步骤3xxxx"]
            }}

            规则：
            1. 只输出 JSON
            2. 不要加 ```json 标记
            3. 如果涉及图片，必须先视觉识别
            4. 如果涉及评估，必须查阅规范
          """),
        ("user", "{input}")
    ])
    
    # 使用本地小模型进行规划，并要求输出结构化数据
    planner = prompt | get_cloud_llm("xunfei_qwen").with_structured_output(Plan)
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
【最高指令：严禁重复调用工具】
1. 视觉识别工具只允许调用【1次】。
2. 规范检索工具（RAG）只需调用【1次】。如果有多种病害（如横向和纵向裂缝），请在一次查询中合并提问，绝不允许为每个裂缝单独调用一次！
3. 只要工具返回了成功的数据（Status: success），你必须立刻停止调用任何工具，并直接输出“数据获取完毕，交由总工生成报告”。
"""),
        ("system", "当前的执行计划是：\n{plan}"),
        MessagesPlaceholder("messages"),
    ])
    return prompt | llm.bind_tools(tools)


def _make_agent_decision(llm, tools):
    """工厂函数：创建绑定 llm/tools 的 agent 决策节点"""
    def agent_decision(state: RoadDiseaseState):
        messages = state["messages"]
        current_plan = "\n".join(state.get("plan", []))
        log.info("agent 节点执行，依据计划执行任务")
        
        _agent = create_agent(llm, tools)
        result = _agent.invoke({"messages": messages, "plan": current_plan})
        return {"messages": result}
    
    return agent_decision


def _get_max_messages() -> int:
    """从配置获取熔断阈值"""
    from config import load_config
    cfg = load_config()
    return cfg.get("agent", {}).get("max_message_count", 12)


def route(state: RoadDiseaseState) -> Literal["tools", "reporter"]:
    max_messages = _get_max_messages()
    
    messages = state.get("messages", [])
    if not messages:
        return "reporter"
    
    # 死循环熔断机制：消息过多说明陷入循环
    if len(messages) > max_messages:
        log.warning(f"⚠️ 触发熔断机制：消息数 {len(messages)} > {max_messages}，强制进入 reporter 节点")
        return "reporter"
        
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        log.info("路由到 tools 节点")
        return "tools"
        
    log.info("路由到 reporter 节点")
    return "reporter"


def build_graph(need_draw: bool = False):
    """构建 LangGraph 工作流，所有重型资源在此函数内延迟初始化"""
    log.info("正在构建工作流图...")
    
    # 延迟初始化：仅在 build_graph 时创建资源实例
    tool_list = _get_tool_list()
    memory = MemorySaver()
    cloud_llm = get_cloud_llm()
    
    # 创建工具节点（使用惰性工具列表）
    tools_node = ToolNode(tool_list)
    
    # 创建绑定 llm/tools 的 agent 决策节点
    agent_node_fn = _make_agent_decision(cloud_llm, tool_list)
    
    builder = StateGraph(RoadDiseaseState)
    
    # 注册节点
    builder.add_node("planner", planner_node)
    builder.add_node("agent", agent_node_fn)
    builder.add_node("reporter", reporter_node)
    builder.add_node("tools", tools_node)
    
    
    # 编排流程
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", "reporter": "reporter"})
    builder.add_edge("reporter", END)
    
    graph = builder.compile(checkpointer=memory)
    if need_draw:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_bytes)
        log.info("已生成流程图: workflow.png")
    
    log.info("工作流图构建完成")
    return graph

if __name__ == "__main__":
  build_graph(True)