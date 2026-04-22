from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RoadDiseaseState(TypedDict):
    # 记录大模型的思考与调用历史
    messages: Annotated[list[BaseMessage], add_messages]

    # Agent 最终生成的管养决策方案
    maintenance_plan: str

    # 记录 Agent 制定的执行步骤计划
    plan: List[str]
