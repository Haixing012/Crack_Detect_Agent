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

@lru_cache(maxsize=1)
def load_config() -> dict:
    """加载并缓存 YAML 配置"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("未找到 config.yaml 文件")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@lru_cache(maxsize=1)
def get_cloud_llm():
    """获取云端大模型，用于复杂推理和工具调用"""
    cfg = load_config()["llm"]["cloud"]
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
    persist_dir = Path(cfg["paths"]["chroma_dir"]).resolve()
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
    model_path = Path(cfg["paths"]["yolo_model"]).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")
    log.info("加载 YOLO 模型: %s", model_path)
    return YOLO(str(model_path))