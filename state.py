from typing import TypedDict, Annotated,List,Dict,Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class DiseaseDetection(TypedDict):
    disease_type: str    # 病害分类，例如："横向裂缝", "坑槽"
    confidence: float    # YOLO视觉检测的置信度
    bbox: List[int]      # 目标边界框坐标 [x_min, y_min, x_max, y_max]
    severity: Optional[str]        # 初始为空，查阅RAG后填入
    dimensions: Dict[str, float]  # 实际物理尺寸估算，例如：{"length_cm": 15.5, "width_cm": 1.2, "depth_cm": 0.0}

class RoadDiseaseState(TypedDict):
    # 记录大模型的思考与调用历史
    messages: Annotated[list[BaseMessage], add_messages]

    # 视觉感知层数据
    image_path: str  # 当前处理的道路图像或视频帧路径
    detection_results: List[DiseaseDetection]  # 存放YOLO/DCNv4识别出的病害类型、位置与严重程度

    # 知识与决策层数据
    retrieved_docs: str  # RAG检索到的对应养护规范与标准
    maintenance_plan: str  # Agent最终生成的管养决策方案

