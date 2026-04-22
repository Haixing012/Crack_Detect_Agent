import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph import build_graph
from logger import get_logger

log = get_logger(__name__)

# ============================================================
# 页面全局配置 & 自定义样式
# ============================================================
PAGE_CONFIG = dict(
    page_title="路面病害智能分析 Agent",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- 全局字体 & 文字颜色（关键：确保所有文字可读）---- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #1e3a5f !important;
}
h1, h2, h3, h4 { color: #0c2d5c !important; }
p, span, label, caption, div { color: #334155 !important; }

/* ---- 隐藏 Streamlit 默认菜单栏/页脚 ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---- 主容器背景：浅色天空蓝渐变 ---- */
.stApp {
    background: linear-gradient(160deg, #e0f2fe 0%, #bae6fd 30%, #e0f7fa 60%, #f0f9ff 100%);
}

/* ---- 标题区卡片：天蓝渐变 ---- */
.title-card {
    background: linear-gradient(120deg, #0ea5e9, #0284c7, #0369a1);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(14,165,233,0.25), 0 2px 8px rgba(0,0,0,0.08);
}
.title-card h1 {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
}
.title-card p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 14px !important;
    margin-top: 6px !important;
}

/* ---- 侧边栏：浅天蓝底 + 白色卡片 ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 50%, #cffafe 100%) !important;
    border-right: 2px solid rgba(14,165,233,0.2) !important;
}
[data-testid="stSidebar"] [class*="header"] {
    background: transparent !important;
}
.sidebar-section {
    background: rgba(255,255,255,0.85);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    border: 1px solid rgba(14,165,233,0.2);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.sidebar-section h3 {
    color: #0369a1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- 上传图片预览 ---- */
.preview-img {
    border-radius: 10px;
    border: 2px solid rgba(14,165,233,0.35);
    box-shadow: 0 4px 16px rgba(14,165,233,0.15);
}
.preview-path {
    font-size: 11px !important;
    color: #475569 !important;
    word-break: break-all;
    background: rgba(224,242,254,0.9);
    padding: 6px 10px;
    border-radius: 6px;
    margin-top: 8px;
    border: 1px solid rgba(14,165,233,0.15);
}

/* ---- 聊天容器 ---- */
.chat-container {
    background: rgba(255,255,255,0.55);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(14,165,233,0.15);
}

/* ---- 用户消息气泡：深蓝色 ---- */
[data-testid="stChatMessage"]:nth-child(odd) {
    background: linear-gradient(135deg, #dbeafe, #e0f2fe) !important;
    border-radius: 14px 14px 4px 14px !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    box-shadow: 0 2px 10px rgba(14,165,233,0.12) !important;
}
[data-testid="stChatMessage"]:nth-child(odd) * { color: #1e3a5f !important; }

/* ---- AI 消息气泡：白色偏蓝 ---- */
[data-testid="stChatMessage"]:nth-child(even) {
    background: linear-gradient(135deg, #ffffff, #f0f9ff) !important;
    border-radius: 14px 14px 14px 4px !important;
    border: 1px solid rgba(14,165,233,0.18) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatMessage"]:nth-child(even) * { color: #334155 !important; }

/* ---- 状态栏美化 ---- */
[data-testid="stStatusWidget"] {
    background: rgba(240,249,255,0.95) !important;
    border: 1px solid rgba(14,165,233,0.3) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 16px rgba(14,165,233,0.1);
}
[data-testid="stStatusWidget"] h3 {
    color: #0369a1 !important;
}
[data-testid="stStatusWidget"] p {
    color: #475569 !important;
    font-size: 13px !important;
}

/* ---- 输入框：白色底 + 蓝色边框，深色文字 ---- */
[data-testid="stTextInput"] > div > div > input,
[data-testid="stTextInput"] textarea {
    background: rgba(255,255,255,0.92) !important;
    border: 1.5px solid rgba(14,165,233,0.35) !important;
    border-radius: 10px !important;
    color: #1e3a5f !important;
    font-size: 14px !important;
    caret-color: #0284c7 !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder,
[data-testid="stTextInput"] textarea::placeholder {
    color: #94a3b8 !important;
}
[data-testid="stTextInput"] > div:focus-within {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
}

/* ---- 按钮：天蓝主色调 ---- */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(14,165,233,0.3) !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important;
}
.stButton button:not([kind="primary"]) {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(14,165,233,0.3) !important;
    border-radius: 8px !important;
    color: #0369a1 !important;
    font-weight: 500 !important;
}
.stButton button:not([kind="primary"]):hover {
    background: #e0f2fe !important;
    border-color: #0ea5e9 !important;
}

/* ---- 文件上传器 ---- */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7) !important;
    border: 2px dashed rgba(14,165,233,0.4) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] label {
    color: #0369a1 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] span {
    color: #64748b !important;
}

/* ---- 滚动条：天蓝色调 ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(14,165,233,0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(14,165,233,0.5); }

/* ---- 装饰标签：浅蓝底 + 深蓝字 ---- */
.tag {
    display: inline-block;
    background: rgba(186,230,253,0.7);
    color: #0369a1;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid rgba(14,165,233,0.2);
}
</style>
"""

# ============================================================
# 常量 / 工具函数
# ============================================================
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def convert_chat_history_to_messages(chat_history: list) -> list:
    """将 (role, content) 元组历史转换为 BaseMessage 列表，供 LangGraph 使用"""
    messages = []
    for role, content in chat_history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    file_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path.resolve())


def init_session() -> None:
    if "app" not in st.session_state:
        st.session_state.app = build_graph()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit_{uuid4().hex}"
    if "uploaded_file_key" not in st.session_state:
        st.session_state.uploaded_file_key = ""
    if "uploaded_image_path" not in st.session_state:
        st.session_state.uploaded_image_path = ""


# ============================================================
# 渲染组件
# ============================================================

def _render_header():
    """顶部标题区"""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="title-card">'
        '<h1>🛣️ 路面病害智能分析 Agent</h1>'
        '<p>基于 YOLO 视觉识别 · ChromaDB RAG 规范检索 · LangGraph 多节点智能编排</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_sidebar() -> str:
    """左侧边栏 → 图片上传 + 会话管理，返回当前图片路径"""
    with st.sidebar:
        # --- Logo / 标题 ---
        st.markdown("### 🧭 控制面板")
        st.markdown("---")

        # --- 图片上传区 ---
        with st.container():
            st.markdown(
                '<div class="sidebar-section">'
                '<h3>📷 图像输入</h3>'
                '</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "上传路面检测图片（支持 JPG/PNG/BMP）",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )

            image_path = st.session_state.uploaded_image_path
            if uploaded_file is not None:
                file_key = f"{uploaded_file.name}:{uploaded_file.size}"
                if file_key != st.session_state.uploaded_file_key:
                    image_path = save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_file_key = file_key
                    st.session_state.uploaded_image_path = image_path
                st.image(
                    uploaded_file,
                    caption=f"预览 · {uploaded_file.name}",
                    use_container_width=True,
                )
                st.markdown(f'<div class="preview-path">📂 {image_path}</div>', unsafe_allow_html=True)
            else:
                st.session_state.uploaded_file_key = ""
                st.session_state.uploaded_image_path = ""
                image_path = ""

        st.markdown("---")

        # --- 操作按钮区 ---
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("🗑️ 清空会话", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.thread_id = f"streamlit_{uuid4().hex}"
                st.session_state.uploaded_file_key = ""
                st.session_state.uploaded_image_path = ""
                st.rerun()
        with col_right:
            if st.button("♻️ 重置 Agent", use_container_width=True):
                st.session_state.app = build_graph()
                st.session_state.chat_history = []
                st.session_state.thread_id = f"streamlit_{uuid4().hex}"
                st.success("Agent 已重新初始化")
                st.rerun()

        # --- 统计信息 ---
        st.markdown("---")
        st.caption(
            f'<div class="tag">对话轮次: {len(st.session_state.get("chat_history", [])) // 2}</div>'
            '&nbsp;'
            f'<div class="tag">会话ID: {st.session_state.get("thread_id", "-")[:8]}</div>',
            unsafe_allow_html=True,
        )

    return image_path


def render_history() -> None:
    """渲染历史消息"""
    for role, content in st.session_state.chat_history:
        display_role = "user" if role == "human" else "assistant"
        with st.chat_message(display_role):
            st.markdown(content)


def ask_agent_stream(user_text: str, image_path: str = ""):
    """处理 Agent 逻辑，返回用于流式输出的生成器，并在状态栏展示节点详情"""
    final_user_text = user_text.strip()
    if image_path:
        final_user_text += (
            f"\n\n请结合以下图片路径进行识别与分析：{image_path}"
            "\n如需要，请调用视觉识别工具。"
        )

    st.session_state.chat_history.append(("human", final_user_text))
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    answer = "未获得有效回复。"

    try:
        with st.status("**🔄 正在分析...**", expanded=True) as status:
            messages_input = convert_chat_history_to_messages(st.session_state.chat_history)
            for chunk in st.session_state.app.stream(
                {"messages": messages_input},
                config=config,
            ):
                for node_name, state in chunk.items():
                    # ==================== Planner 节点 ====================
                    if node_name == "planner":
                        st.markdown("#### 📋 任务规划")
                        plan_steps = state.get("plan", [])
                        if plan_steps:
                            for i, step in enumerate(plan_steps, 1):
                                st.markdown(f"**步骤 {i}** — {step}")
                        else:
                            st.caption("*（未生成计划步骤）*")
                        st.divider()

                    # ==================== Agent 节点 ====================
                    elif node_name == "agent":
                        st.markdown("#### 🧠 Agent 决策")
                        raw_msgs = state.get("messages", [])
                        # agent 节点可能返回单个 AIMessage 或列表，统一处理
                        msgs = raw_msgs if isinstance(raw_msgs, list) else [raw_msgs]
                        if msgs:
                            last_msg = msgs[-1]
                            # 检查是否有工具调用
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                st.markdown("**决策结果：调用以下工具**")
                                for tc in last_msg.tool_calls:
                                    tool_name = tc.get("name", "未知工具")
                                    tool_args = tc.get("args", {})
                                    args_str = ", ".join(
                                        f"{k}=`{v}`" for k, v in tool_args.items()
                                    )
                                    st.info(f"🔧 **{tool_name}**\n\n参数: {args_str}")
                            else:
                                content = getattr(last_msg, "content", "")
                                st.markdown(f"**Agent 思考**: {content[:200]}{'...' if len(content) > 200 else ''}")
                        else:
                            st.caption("*Agent 无输出*")
                        st.divider()

                    # ==================== Tools 节点 ====================
                    elif node_name == "tools":
                        st.markdown("#### 🔧 工具执行结果")
                        raw_msgs = state.get("messages", [])
                        msgs = raw_msgs if isinstance(raw_msgs, list) else [raw_msgs]
                        for msg in msgs:
                            # 工具返回的消息有 name 属性标识来源
                            tool_name = getattr(msg, "name", None)
                            content = getattr(msg, "content", "")

                            if tool_name:
                                # 根据不同工具用不同样式展示
                                display_name_map = {
                                    "predict_image_crack": "🔍 YOLO 裂缝识别",
                                    "get_retrieve_docs": "📚 规范检索 (RAG)",
                                }
                                label = display_name_map.get(tool_name, f"🔧 {tool_name}")

                                try:
                                    import json as _json
                                    parsed = _json.loads(content)
                                    detail = _json.dumps(parsed, ensure_ascii=False, indent=2)
                                except Exception:
                                    detail = content[:500]

                                st.success(f"**{label}**")
                                with st.expander("查看详细返回数据"):
                                    st.code(detail, language="json" if tool_name != "predict_image_crack" else None)

                        st.divider()

                    # ==================== Reporter 节点 ====================
                    elif node_name == "reporter":
                        st.markdown("#### 📝 正在生成养护方案报告...")
                        answer = state.get("maintenance_plan", "")
                        if not answer and state.get("messages"):
                            raw_msgs = state["messages"]
                            last_msg = raw_msgs[-1] if isinstance(raw_msgs, list) else raw_msgs
                            answer = getattr(last_msg, "content", "")

            status.update(label="✅ 分析完成！", state="complete", expanded=False)

        st.session_state.chat_history.append(("ai", answer))

    except Exception as e:
        log.error("Agent 执行异常: %s", str(e), exc_info=True)
        error_msg = f"⚠️ 处理异常：`{str(e)}`\n\n请检查日志或重试。"
        st.session_state.chat_history.append(("ai", error_msg))

        def error_generator():
            for ch in error_msg:
                yield ch
                time.sleep(0.01)
        return error_generator()

    def type_writer():
        chunk_size = 3
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]
            time.sleep(0.02)

    return type_writer()


# ============================================================
# 入口
# ============================================================

def main() -> None:
    st.set_page_config(**PAGE_CONFIG)

    _render_header()
    init_session()
    image_path = _render_sidebar()

    # 分隔线 + 提示语
    st.markdown("### 💬 对话窗口")
    st.caption(
        "输入您的问题，例如：*「这条裂缝的严重等级？应采取什么养护措施？」*\n"
        "上传图片后可自动触发 YOLO 视觉检测。"
    )

    render_history()

    prompt = st.chat_input("请输入问题...")
    if prompt:
        user_display = prompt
        if image_path:
            user_display += "\n\n📎 *已附带上传图片*"

        with st.chat_message("user"):
            st.markdown(user_display)

        with st.chat_message("assistant"):
            stream_generator = ask_agent_stream(prompt, image_path=image_path)
            st.write_stream(stream_generator)


if __name__ == "__main__":
    main()
