import os
import yaml
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from ultralytics import YOLO

from logger import get_logger

log = get_logger(__name__)

# 获取项目根目录：基于当前脚本所在目录向上定位
_PROJECT_ROOT = Path(__file__).parent.resolve()

@lru_cache(maxsize=1)
def load_config() -> dict:
    """加载并缓存 YAML 配置，路径基于脚本所在目录解析"""
    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}\n请复制 config.example.yaml 为 config.yaml 并填写配置")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@lru_cache(maxsize=1)
def get_cloud_llm(model="deepseek"):
    """获取云端大模型，用于复杂推理和工具调用"""
    cfg = load_config()["llm"]["cloud"][model]
    log.info("初始化云端 LLM: model=%s", cfg["model"])
    return ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=0,
    )

@lru_cache(maxsize=1)
def get_local_llm():
    """获取本地大模型，专门用于 Plan (规划) 节点"""
    cfg = load_config()["llm"]["local"]
    log.info("初始化本地 LLM: model=%s", cfg["model"])
    return ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg.get("api_key", "ollama"),
        model=cfg["model"],
        temperature=0,
    )

@lru_cache(maxsize=1)
def get_chroma():
    cfg = load_config()
    # 支持相对路径（基于项目根目录）和绝对路径
    chroma_dir = cfg["paths"]["chroma_dir"]
    persist_dir = (Path(chroma_dir) if Path(chroma_dir).is_absolute()
                   else _PROJECT_ROOT / chroma_dir).resolve()
    log.info("初始化 Chroma: dir=%s", persist_dir)
    model = OllamaEmbeddings(model=cfg["embedding"]["model"])
    return Chroma(
        embedding_function=model,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

@lru_cache(maxsize=1)
def get_yolo():
    cfg = load_config()
    yolo_path = cfg["paths"]["yolo_model"]
    model_path = (Path(yolo_path) if Path(yolo_path).is_absolute()
                  else _PROJECT_ROOT / yolo_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")
    log.info("加载 YOLO 模型: %s", model_path)
    return YOLO(str(model_path))

if __name__ == "__main__":
    print(get_chroma())