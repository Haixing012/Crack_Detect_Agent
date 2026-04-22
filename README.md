# 路面病害智能分析 Agent（YOLO + RAG + LangGraph）

面向道路病害问答与辅助决策的智能分析系统：
- 使用 **YOLO** 做裂缝/坑槽等目标识别与尺寸估算
- 使用 **ChromaDB + Embedding** 做养护规范知识检索（RAG）
- 使用 **LangGraph** 编排多节点协同 Agent（Planner → Agent → Tools → Reporter）

## 1. 环境要求

- Python 3.10+
- 可用的 LLM API Key（兼容 OpenAI 接口格式，如通义千问、DeepSeek 等）
- 本地模型文件：`./models/11nbase.pt`
- 本地向量库目录：`./chroma_db`

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 配置

项目根目录下已提供配置模板，请复制并填写：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
llm:
  cloud:
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 云端大模型地址
    api_key: "your-api-key"                                           # API 密钥
    model: "qwen-plus"                                                # 云端模型名
  local:
    base_url: "http://localhost:11434/v1"                              # 本地 Ollama 地址
    api_key: "ollama"
    model: "qwen2.5:7b"                                               # 本地规划模型（用于 Planner 节点）

embedding:
  model: "qwen3-embedding:8b"                                         # Ollama 嵌入模型

paths:
  chroma_dir: "./chroma_db"
  yolo_model: "./models/11nbase.pt"

agent:
  max_message_count: 12                                                # 死循环熔断阈值

gsd:                                                                   # 无人机 GSD 参数（影响尺寸换算精度）
  altitude_m: 5.0
  focal_length_mm: 4.5
  sensor_width_mm: 6.4
  image_width_px: 640
```

> 说明：
> - `config.py` 启动时基于脚本所在目录自动定位 `config.yaml`，不受工作目录影响
> - 若 `api_key` 为空或文件缺失，启动时会抛出明确错误提示
> - 本地模型（local）仅用于 Planner 节点的任务拆解，可使用轻量级模型

## 4. 启动方式

### CLI 命令行模式

```bash
python run.py
```

启动后输入自然语言问题，输入 `q` 退出。

### Streamlit Web UI 模式

```bash
streamlit run streamlit_app.py --server.port 8501
```

浏览器打开 `http://localhost:8501` 即可使用。

## 5. 项目结构

```
cli/
├── config.py              # YAML 配置加载，LLM/YOLO/Chroma 延迟初始化工厂
├── graph.py               # LangGraph 工作流编排（Planner→Agent→Tools→Reporter）
├── state.py               # Graph 状态 TypedDict 定义
├── tools.py               # YOLO 识别工具 + RAG 检索工具定义（惰性加载）
├── rag.py                 # ChromaDB 向量检索封装（惰性加载）
├── utils.py               # 像素到真实尺寸(GSD)换算工具类
├── logger.py              # 统一日志初始化
├── run.py                 # CLI 入口，维护对话循环
├── streamlit_app.py       # Streamlit Web UI
├── config.example.yaml    # 配置模板（含所有可用字段注释）
├── config.yaml            # 实际配置文件（需自行创建）
├── requirements.txt       # Python 依赖清单
└── models/
    └── 11nbase.pt         # YOLO 训练权重文件
```

## 6. 架构说明

### 工作流节点

| 节点 | 职责 | 使用的模型 |
|------|------|-----------|
| **Planner** | 将用户问题拆解为结构化执行步骤 | 本地轻量 LLM |
| **Agent** | 按计划调度工具（视觉识别 / RAG 检索） | 云端大模型 |
| **Tools** | 执行 YOLO 检测 或 ChromaDB 检索 | YOLO / Embedding |
| **Reporter** | 基于客观数据生成专业养护方案报告 | 云端大模型 |

### 安全机制

- **死循环熔断**: 当消息轮数超过 `agent.max_message_count`（默认12）时强制进入 Reporter 节点
- **工具调用限制**: Agent 提示词中严格限制每种工具只允许调用一次
- **异常捕获**: Streamlit 前端对图执行过程做了完整 try-except，避免崩溃

### 延迟初始化策略

所有重型资源（YOLO 模型、向量库、LLM 连接）均采用惰性加载：
- `import graph.py` **不会**触发任何模型加载
- 仅在 `build_graph()` 首次被调用时才创建全部实例
- 这使得单元测试和导入速度大幅改善

## 7. 日志

日志格式：

```
时间 | 级别 | 模块 | 消息
```

默认输出到控制台，通过 `config.yaml` 可扩展为文件输出。

## 8. License

MIT
