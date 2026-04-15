from graph import build_graph
from logger import get_logger


log = get_logger(__name__)


def main() -> None:
    app = build_graph()
    chat_history = []

    print("对话已开始，输入 'q' 退出。")
    log.info("应用启动成功，等待用户输入")

    while True:
        user_input = input("\nHuman: ")
        if user_input.strip().lower() == "q":
            print("对话结束。")
            log.info("用户主动退出")
            break

        chat_history.append(("human", user_input))
        log.info("收到用户输入，当前历史消息数: %s", len(chat_history))

        try:
            for chunk in app.stream({"messages": chat_history}):
                for node_name, state in chunk.items():
                    msg = state.get("messages")
                    if msg and hasattr(msg, "content"):
                        print(f"[{node_name}]: {msg.content}")
                        print("-" * 30)
        except Exception:
            log.exception("图执行失败")
            print("执行失败，请查看日志。")


if __name__ == "__main__":
    main()
