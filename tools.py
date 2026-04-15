from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import get_yolo
from logger import get_logger
from rag import retrieve_docs
from utils import PixelToRealConverter


log = get_logger(__name__)
yolo_model = get_yolo()


@tool(description="使用YOLO模型识别图像中的裂缝信息。")
def predict_image_crack(image_path: str) -> str:
    """
    识别图片中的裂缝信息。
    参数:
    image_path: 图片文件路径
    返回:
    str: 包含裂缝分类、定位和尺寸信息的格式化字符串
    """
    try:
        log.info("开始裂缝识别: %s", image_path)
        results = yolo_model(image_path)
        class_names = yolo_model.names if hasattr(yolo_model, "names") else {}

        if not results:
            return "未检测到裂缝"

        result = results[0]
        output = []
        converter = PixelToRealConverter()
        gsd = converter.calculate_gsd_by_drone(
            altitude_m=5.0,
            focal_length_mm=4.5,
            sensor_width_mm=6.4,
            image_width_px=640,
        )

        for i, box in enumerate(result.boxes):
            x_center, y_center, width, height = box.xywh[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = class_names.get(class_id, f"类别{class_id}")

            real_width_cm = converter.convert(width, gsd) / 10
            real_height_cm = converter.convert(height, gsd) / 10

            output.append(
                f"裂缝{i + 1} [{class_name}]: "
                f"中心坐标({x_center:.1f}, {y_center:.1f}), "
                f"尺寸{real_width_cm:.1f}cmx{real_height_cm:.1f}cm, "
                f"置信度{confidence:.2f}"
            )

        if not output:
            return "未检测到裂缝"

        log.info("裂缝识别完成，目标数: %s", len(output))
        return "检测结果:\n" + "\n".join(output) + "\n请立即调用RAG检索。"

    except FileNotFoundError as e:
        log.exception("图片不存在: %s", image_path)
        return f"错误: 文件未找到 {str(e)}"
    except Exception as e:
        log.exception("裂缝识别异常")
        return f"预测过程中发生错误: {str(e)}"


class RAGInput(BaseModel):
    disease_type: str = Field(description="病害类型，如横向裂缝")
    width_cm: float = Field(0.0, description="宽度")
    length_cm: float = Field(0.0, description="长度")


@tool(
    description="""
当你获取到病害类型和尺寸时，必须调用此工具。
当需要评估病害严重等级或获取标准养护规范时，必须调用此工具。
""",
    args_schema=RAGInput,
)
def get_retrieve_docs(
    disease_type: str, width_cm: float = 0.0, length_cm: float = 0.0
) -> str:
    """当需要评估病害严重等级或获取标准养护规范时，必须调用此工具。"""
    query = f"""
请根据病害类型和尺寸，提供病害严重等级和标准养护规范。
病害类型: {disease_type},
宽度: {width_cm}cm,
长度: {length_cm}cm
"""
    log.info("执行结构化 RAG 检索: disease_type=%s", disease_type)
    return retrieve_docs(query, k=3, score_threshold=0.3)


@tool(
    description="""
当用户提问道路病害相关信息时必须调用此工具查询《公路技术状况评定标准》。
当需要评估病害严重等级或获取标准养护规范时，必须调用此工具。
"""
)
def query_retrieve_docs(query: str) -> str:
    log.info("执行自由查询 RAG 检索")
    return retrieve_docs(query, k=3, score_threshold=0.3)


if __name__ == "__main__":
    pass
