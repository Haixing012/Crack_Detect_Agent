# 路面病害智能分析（YOLO + RAG + LangGraph）

一个面向道路病害问答与辅助决策的最小可运行项目：
- 使用 `YOLO` 做裂缝目标识别
- 使用 `Chroma + Embedding` 做知识检索（RAG）
- 使用 `LangGraph` 编排 Agent 与工具调用

## 1. 环境要求

- Python 3.9+
- 可用的 LLM API Key（用于 `ChatOpenAI` 兼容接口）
- 本地模型文件：`./models/11nbase.pt`
- 本地向量库目录：`./chroma_db`

## 2. 安装依赖

示例（按当前代码需要）：

```bash
pip install langchain langgraph langchain-openai langchain-ollama langchain-chroma ultralytics python-dotenv
```

## 3. 配置 `.env`

项目根目录下已提供 `.env` 模板，请至少填写 `LLM_API_KEY`：

```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的key
LLM_MODEL=qwen3.5-plus
EMBEDDING_MODEL=qwen3-embedding:8b
CHROMA_DIR=./chroma_db
YOLO_MODEL_PATH=./models/11nbase.pt
LOG_LEVEL=INFO
```

说明：
- `config.py` 启动时会自动读取 `.env`
- 若 `LLM_API_KEY` 为空，会抛出明确错误

## 4. 启动

```bash
python run.py
```

启动后输入自然语言问题，输入 `q` 退出。

## 5. 项目结构

- `run.py`：CLI 入口，维护对话循环
- `graph.py`：LangGraph 编排（Agent/Tools 路由）
- `tools.py`：YOLO 识别与 RAG 查询工具
- `rag.py`：向量检索封装
- `config.py`：配置加载与模型初始化
- `logger.py`：统一日志初始化
- `state.py`：Graph 状态定义
- `utils.py`：像素到真实尺寸换算

## 6. 日志

日志格式：

```text
时间 | 级别 | 模块 | 消息
```

通过 `.env` 中 `LOG_LEVEL` 控制级别，例如 `DEBUG/INFO/WARNING/ERROR`。


