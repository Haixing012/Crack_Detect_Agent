import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from ultralytics import YOLO

from logger import get_logger


log = get_logger(__name__)


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3.5-plus"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
    )
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_db"))
    yolo_model_path: str = field(
        default_factory=lambda: os.getenv("YOLO_MODEL_PATH", "./models/11nbase.pt")
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()


@lru_cache(maxsize=1)
def get_llm():
    cfg = get_config()
    if not cfg.llm_api_key:
        raise ValueError("缺少 LLM_API_KEY 或 OPENAI_API_KEY 环境变量，请检查 .env")
    log.info("初始化 LLM: model=%s", cfg.llm_model)
    return ChatOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_chroma():
    cfg = get_config()
    persist_dir = Path(cfg.chroma_dir).resolve()
    log.info("初始化 Chroma: dir=%s", persist_dir)
    model = OllamaEmbeddings(model=cfg.embedding_model)
    return Chroma(
        embedding_function=model,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def get_yolo():
    cfg = get_config()
    model_path = Path(cfg.yolo_model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")
    log.info("加载 YOLO 模型: %s", model_path)
    return YOLO(str(model_path))
