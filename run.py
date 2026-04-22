from langchain_core.messages import HumanMessage, AIMessage

from graph import build_graph
from logger import get_logger
from config import get_cloud_llm,get_local_llm

log = get_logger(__name__)


def convert_chat_history_to_messages(chat_history: list) -> list:
    """将 (role, content) 元组历史转换为 BaseMessage 列表"""
    messages = []
    for role, content in chat_history:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def main() -> None:
    # 启动时主动预热，提前将模型加载到内存
    get_local_llm()
    get_cloud_llm()
    app = build_graph()
    chat_history = []

    print("对话已开始，输入 'q' 退出。")
    log.info("应用启动成功，等待用户输入")
    config = {
        "configurable": {
            "thread_id": "session_001"
        }
    }

    while True:
        user_input = input("\nHuman: ")
        if user_input.strip().lower() == "q":
            print("对话结束。")
            log.info("用户主动退出")
            break

        chat_history.append(("human", user_input))
        log.info("收到用户输入，当前历史消息数: %s", len(chat_history))

        try:
            messages_input = convert_chat_history_to_messages(chat_history)
            for chunk in app.stream({"messages": messages_input},config=config):
                for node_name, state in chunk.items():
                    msgs = state.get("messages", [])
                    if isinstance(msgs, list) and len(msgs) > 0:
                        # 取最后一条有内容的消息
                        for m in reversed(msgs):
                            if hasattr(m, "content") and m.content and not hasattr(m, "tool_calls"):
                                print(f"[{node_name}]: {m.content}")
                                print("-" * 30)
                                break
        except Exception:
            log.exception("图执行失败")
            print("执行失败，请查看日志。")


if __name__ == "__main__":
    main()
