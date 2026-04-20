from pathlib import Path
from uuid import uuid4

import streamlit as st

from graph import build_graph


UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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


def render_history() -> None:
    for role, content in st.session_state.chat_history:
        display_role = "user" if role == "human" else "assistant"
        with st.chat_message(display_role):
            st.markdown(content)


def ask_agent(user_text: str, image_path: str = "") -> str:
    final_user_text = user_text.strip()
    if image_path:
        final_user_text += (
            f"\n\n请结合以下图片路径进行识别与分析：{image_path}"
            "\n如需要，请调用视觉识别工具。"
        )

    st.session_state.chat_history.append(("human", final_user_text))
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = st.session_state.app.invoke(
        {"messages": st.session_state.chat_history},
        config=config,
    )
    messages = result.get("messages", [])
    answer = "未获得有效回复。"
    if messages and hasattr(messages[-1], "content"):
        answer = messages[-1].content
    st.session_state.chat_history.append(("ai", answer))
    return answer


def main() -> None:
    st.set_page_config(page_title="路面病害智能问答", layout="wide")
    st.title("路面病害智能问答")
    st.caption("基于 YOLO + RAG + LangGraph 的 Streamlit 界面")

    init_session()

    with st.sidebar:
        st.subheader("图片输入")
        uploaded_file = st.file_uploader(
            "上传路面图片（可选）",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        image_path = st.session_state.uploaded_image_path
        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}:{uploaded_file.size}"
            if file_key != st.session_state.uploaded_file_key:
                image_path = save_uploaded_file(uploaded_file)
                st.session_state.uploaded_file_key = file_key
                st.session_state.uploaded_image_path = image_path
            st.image(uploaded_file, caption="当前上传图片", use_container_width=True)
            st.caption(f"已保存路径: {image_path}")
        else:
            st.session_state.uploaded_file_key = ""
            st.session_state.uploaded_image_path = ""
            image_path = ""

        if st.button("清空会话"):
            st.session_state.chat_history = []
            st.session_state.thread_id = f"streamlit_{uuid4().hex}"
            st.session_state.uploaded_file_key = ""
            st.session_state.uploaded_image_path = ""
            st.rerun()

    render_history()

    prompt = st.chat_input("请输入问题，例如：这条裂缝该如何养护？")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt if not image_path else f"{prompt}\n\n(已附带上传图片)")

        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                answer = ask_agent(prompt, image_path=image_path)
                st.markdown(answer)


if __name__ == "__main__":
    main()
