from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List,Optional
from utils import PixelToRealConverter

import json

log = None
_yolo_model = None


def _get_yolo_model():
    """惰性加载 YOLO 模型，仅在首次工具调用时初始化"""
    global _yolo_model, log
    if _yolo_model is None:
        from config import get_yolo
        from logger import get_logger
        if log is None:
            log = get_logger(__name__)
        _yolo_model = get_yolo()
    return _yolo_model


@tool(description="使用YOLO模型识别图像中的裂缝信息。")
def predict_image_crack(image_path: str) -> json:
    """
    识别图片中的裂缝信息。
    参数:
    image_path: 图片文件路径
    返回:
    json: 包含裂缝分类、定位和尺寸信息
    """
    try:
        _log = log or __import__("logger").get_logger(__name__)
        _log.info("开始裂缝识别: %s", image_path)
        yolo_model = _get_yolo_model()
        results = yolo_model(image_path)
        class_names = yolo_model.names if hasattr(yolo_model, "names") else {}

        if not results:
            return "未检测到裂缝"

        result = results[0]
        output = []
        converter = PixelToRealConverter()

        # 从配置文件读取 GSD 参数，避免硬编码
        _cfg = __import__("config").load_config()
        gsd_cfg = _cfg.get("gsd", {})
        gsd = converter.calculate_gsd_by_drone(
            altitude_m=gsd_cfg.get("altitude_m", 5.0),
            focal_length_mm=gsd_cfg.get("focal_length_mm", 4.5),
            sensor_width_mm=gsd_cfg.get("sensor_width_mm", 6.4),
            image_width_px=gsd_cfg.get("image_width_px", 640),
        )

        for i, box in enumerate(result.boxes):
            x_center, y_center, width, height = box.xywh[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = class_names.get(class_id, f"类别{class_id}")

            real_width_cm = converter.convert(width, gsd) / 10
            real_height_cm = converter.convert(height, gsd) / 10

            output.append({
                "id": i + 1,
                "disease_type": class_name,
                "confidence": round(confidence, 2),
                "dimensions": {
                    "width_cm": real_width_cm,
                    "length_cm": real_height_cm
                }
            })

        if not output:
            return json.dumps({"status": "success", "data": [], "message": "未检测到裂缝"}, ensure_ascii=False)

        _log = log or __import__("logger").get_logger(__name__)
        _log.info("裂缝识别完成，目标数: %s", len(output))
        return json.dumps({"status": "success", "data": output}, ensure_ascii=False)

    except FileNotFoundError as e:
        _log = log or __import__("logger").get_logger(__name__)
        _log.exception("图片不存在: %s", image_path)
        return f"错误: 文件未找到 {str(e)}"
    except Exception as e:
        _log = log or __import__("logger").get_logger(__name__)
        _log.exception("裂缝识别异常")
        return f"预测过程中发生错误: {str(e)}"


class RAGInput(BaseModel):
    disease_type: str = Field(description="病害类型，如横向裂缝")
    width_cm: float = Field(0.0, description="宽度")
    length_cm: float = Field(0.0, description="长度")

class DocumentDetail(BaseModel):
    source: str = Field(description="规范文档名称或来源")
    content: str = Field(description="具体的规范条文内容")
    score: float = Field(description="检索匹配相关度评分")

class RAGResponse(BaseModel):
    status: str = Field(description="执行状态：success 或 failed")
    data: List[DocumentDetail] = Field(default_factory=list, description="检索到的规范详情列表")
    message: Optional[str] = Field(None, description="错误信息或提示")


@tool(
    description="评估病害严重等级或获取标准养护规范。",
    args_schema=RAGInput,
)
def get_retrieve_docs(
    disease_type: str, width_cm: float = 0.0, length_cm: float = 0.0
) -> str:
    """当获取到病害尺寸后，调用此工具获取对应的养护标准。"""
    from rag import retrieve_docs  # 延迟导入避免模块级依赖
    query = f"病害类型: {disease_type}, 尺寸: 宽{width_cm}cm, 长{length_cm}cm 的养护分级和措施"
    _log = log or __import__("logger").get_logger(__name__)
    _log.info("执行结构化 RAG 检索")
    
    try:
        docs = retrieve_docs(query)
        if not docs:
            return RAGResponse(status="failed", message="库中未找到匹配的养护规范").model_dump_json()
        
        # 返回标准的 JSON 字符串
        return RAGResponse(status="success", data=docs).model_dump_json()
    except Exception as e:
        return RAGResponse(status="error", message=str(e)).model_dump_json()

if __name__ == "__main__":
    pass
