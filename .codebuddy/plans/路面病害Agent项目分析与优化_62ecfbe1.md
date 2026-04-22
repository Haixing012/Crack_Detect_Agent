---
name: 路面病害Agent项目分析与优化
overview: 对当前路面病害智能分析Agent项目进行全面问题诊断与优化改进，覆盖架构设计、代码质量、健壮性、性能等多个维度。
design:
  architecture:
    framework: react
    component: tdesign
  styleKeywords:
    - Glassmorphism
    - 科技感深色主题
    - 毛玻璃卡片
    - 流畅微交互动画
    - 渐进式状态指示
  fontSystem:
    fontFamily: PingFang SC
    heading:
      size: 24px
      weight: 600
    subheading:
      size: 16px
      weight: 500
    body:
      size: 14px
      weight: 400
  colorSystem:
    primary:
      - "#3B82F6"
      - "#2563EB"
      - "#1D4ED8"
    background:
      - "#0F172A"
      - "#1E293B"
      - "#334155"
    text:
      - "#F8FAFC"
      - "#CBD5E1"
      - "#94A3B8"
    functional:
      - "#10B981"
      - "#EF4444"
      - "#F59E0B"
todos:
  - id: fix-critical-bugs
    content: 修复 run.py 消息列表遍历Bug 和 streamlit_app.py 数据格式不匹配问题
    status: completed
  - id: refactor-lazy-init
    content: 重构 graph.py/tools.py/rag.py 移除模块级副作用，改为延迟初始化/惰性加载
    status: completed
    dependencies:
      - fix-critical-bugs
  - id: cleanup-dead-code
    content: 清理 state.py 死代码字段、tools.py 未使用工具、utils.py 冗余构造函数
    status: completed
    dependencies:
      - refactor-lazy-init
  - id: centralize-config
    content: 将 GSD 硬编码参数、熔断阈值迁移至 config.yaml，修正配置路径为基于脚本目录解析
    status: completed
    dependencies:
      - refactor-lazy-init
  - id: enhance-streamlit-ui
    content: 增强 streamlit_app.py 异常处理和UI体验（错误提示、状态可视化）
    status: completed
    dependencies:
      - fix-critical-bugs
      - refactor-lazy-init
  - id: polish-project
    content: 规范化 requirements.txt（补依赖+版本号）、更新 README.md 与代码对齐
    status: completed
    dependencies:
      - centralize-config
      - cleanup-dead-code
  - id: verify-changes
    content: 使用 [subagent:code-explorer] 全面验证重构后的依赖关系、导入链和数据流向
    status: completed
---

## 产品概述

当前项目是一个**路面病害智能分析 Agent**，基于 YOLO(视觉识别) + ChromaDB/RAG(规范检索) + LangGraph(Agent编排) 技术栈，支持 CLI 和 Streamlit 两种交互界面。用户需要全面分析项目存在的问题并实施改进优化。

## 核心问题汇总（按优先级排序）

### 严重级别（必须修复）

1. **模块级副作用 / 导入即重型初始化**：`graph.py`、`tools.py`、`rag.py` 在模块导入时就触发 YOLO 模型加载、向量库初始化、LLM 连接创建，导致无法单元测试、启动极慢、任何 import 都会引发副作用链
2. **run.py 运行时 Bug**：`state["messages"]` 返回的是 message 列表，但代码用 `msg.content` 当作单个消息对象访问，导致输出始终为空或报错
3. **streamlit_app.py 数据格式不匹配**：chat_history 是 `("human"/"ai", content)` 元组列表，但 LangGraph State 期望 `list[BaseMessage]`，类型完全不一致会导致图执行异常

### 中等级别（影响可维护性与功能正确性）

4. **state.py 死代码字段**：`image_path`、`detection_results`、`retrieved_docs` 已定义但从未被任何节点写入
5. **tools.py 死代码工具**：`query_retrieve_docs` 已定义但未注册到 tool_list
6. **config.py 配置路径硬编码**：`Path("config.yaml")` 使用相对路径依赖 CWD，从不同目录启动会找不到文件
7. **tools.py GSD 参数硬编码**：无人机拍摄参数（高度5m、焦距4.5mm等）写死在代码中，应从配置读取或作为参数暴露
8. **requirements.txt 缺失与不规范**：缺少 pyyaml 依赖、所有包无版本锁定、缺少版本范围约束
9. **streamlit_app.py 无异常处理**：`app.stream()` 调用无 try-except，节点报错导致前端崩溃无友好提示

### 低级别（代码质量改进）

10. **utils.py 冗余构造函数**：`PixelToRealConverter.__init__` 为空 pass，所有方法都是 staticmethod
11. **graph.py 魔法数字熔断阈值**：`len(messages) > 12` 硬编码
12. **README.md 与实际代码不一致**：描述使用 .env 配置但实际使用 config.yaml
13. **缺少包结构与 init.py**：扁平脚本结构不利于后续扩展

## 技术栈

- **核心框架**: Python 3.9+ / LangGraph / LangChain Core
- **LLM 接入**: langchain-openai (ChatOpenAI 兼容接口)
- **本地模型**: langchain-ollama (OllamaEmbeddings)
- **视觉识别**: ultralytics (YOLO)
- **向量数据库**: langchain-chroma (ChromaDB)
- **Web UI**: streamlit
- **配置管理**: pyyaml
- **日志**: logging (标准库)

## 实施方案

### 策略概述

采用**延迟初始化 + 工厂模式 + 惰性单例**策略解决重型资源的模块级副作用问题；修复数据格式转换层使 Streamlit 与 LangGraph 正确对接；清理死代码并规范化配置体系。整体保持现有架构不变（Planner -> Agent -> Tools -> Reporter 四节点流程），聚焦于健壮性和可维护性提升。

### 关键技术决策

1. **延迟初始化重构**: 将所有模块级的资源实例化（YOLO模型、ChromaDB、LLM连接、MemorySaver）改为函数内部的惰性加载或工厂方法调用。`build_graph()` 成为唯一的组装入口，外部不暴露裸资源
2. **消息格式适配层**: 在 `streamlit_app.py` 和 `run.py` 中增加元组格式到 BaseMessage 的转换函数，确保输入数据与 State 定义一致
3. **配置中心化**: 所有硬编码参数（GSD参数、熔断阈值等）迁移至 config.yaml 统一管理
4. **分层异常处理**: 图执行外层统一 try-except，区分用户提示错误和系统内部错误

### 架构设计

```
入口层 (run.py / streamlit_app.py)
  ↓ 调用 build_graph() 延迟组装
编排层 (graph.py) -- 仅定义节点函数和边，不持有资源实例
  ↓ 按需获取
资源层 (config.py) -- @lru_cache 惰性单例工厂
  ↓ 依赖
工具层 (tools.py / rag.py) -- 纯函数定义，无模块级副作用
```

### 目录结构变更

```
h:\Agent\git_version\cli\
├── config.py                  # [MODIFY] 配置路径修正 + 新增 GSD/熔断配置项
├── graph.py                   # [MODIFY] 移除模块级副作用，延迟初始化 agent/memory
├── streamlit_app.py           # [MODIFY] 消息格式转换 + 异常处理 + UI增强
├── run.py                     # [MODIFY] 修复 msg 列表遍历 bug + 消息格式转换
├── tools.py                   # [MODIFY] 移除模块级 yolo_model 初始化，GSD参数化
├── rag.py                     # [MODIFY] 移除模块级 vector_store 初始化
├── state.py                   # [MODIFY] 清理死代码字段，添加 Optional 注解
├── utils.py                   # [MODIFY] 移除冗余 __init__，改用类方法/纯函数
├── logger.py                  # [不变]
├── requirements.txt           # [MODIFY] 补全依赖+版本号
├── config.example.yaml        # [MODIFY] 新增 GSD/熔断等配置项
└── README.md                  # [MODIFY] 与实际代码对齐
```

### 关键实现要点

- **graph.py 重构核心**: 将 `tool_list`, `memory`, `cloud_llm`, `agent` 四个模块级变量全部移入 `build_graph()` 函数内部；`create_agent()` 接收 llm 和 tools 参数而非引用全局变量
- **tools.py 安全导入**: `yolo_model = get_yolo()` 移到各工具函数内部或通过闭包注入，避免 import 时加载几百MB的模型文件
- **rag.py 同理处理**: `vector_store = get_chroma()` 移到 `retrieve_docs()` 内部或使用惰性属性
- **消息转换函数**: 新增 `convert_chat_history_to_messages(history)` 工具函数，将 `(role, content)` 元组转为 `HumanMessage/AIMessage`
- **run.py 修复**: `msg = state.get("messages")` 后需迭代 list 取最后一条消息的 `.content`
- **requirements.txt 规范**: 锁定主版本号（如 `langchain-core>=0.3`），补充 `pyyaml>=6.0`

## 设计风格定位

针对路面病害智能分析 Agent 的 Web 界面，采用**科技感 Glassmorphism 风格**，体现 AI + 工程检测的专业性。

## 页面规划

### 主页面（唯一页面）-- 智能问答工作台

整体采用左右分栏布局：左侧为主对话区域（占70%宽度），右侧为辅助面板（占30%宽度）。

#### 顶部导航栏

- 左侧：系统 Logo + "路面病害智能分析 Agent" 标题
- 右侧：模型状态指示器（云端LLM/本地LLM/YOLO/向量库 各一个状态圆点）
- 底色：深色渐变 (#0f172a → #1e293b)

#### 对话区（左侧主区域）

- **历史消息流**: 采用气泡式聊天布局，用户消息靠右（蓝色系），助手消息靠左（深灰玻璃态），支持 Markdown 渲染
- **图片预览卡片**: 当用户上传图片时，在对话流中嵌入缩略图卡片，支持点击放大查看
- **输入区域**: 底部固定输入栏，包含文本输入框 + 图片上传按钮 + 发送按钮，支持拖拽上传

#### 辅助面板（右侧侧边栏）

- **图片上传区块**: 大尺寸上传区域，拖拽或点击上传，显示已上传图片预览及基本信息（文件名、尺寸）
- **执行过程可视化**: 显示当前 Agent 节点执行状态的步骤条（Planner -> Agent -> Tools -> Reporter），每个步骤完成后高亮
- **快捷操作**: "清空会话"、"导出报告"按钮组
- **系统信息**: 当前配置摘要、会话ID等

#### 底部状态栏

- 显示当前线程ID、消息计数、响应耗时

## SubAgent

- **code-explorer**
- Purpose: 深度验证重构后模块间的依赖关系和导入链是否正确
- Expected outcome: 确认修改后的代码不存在循环导入、遗漏依赖或类型不匹配问题