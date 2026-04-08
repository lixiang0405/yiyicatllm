"""
Gradio 聊天界面
连接 vLLM 的 OpenAI-compatible API，提供 Web 交互界面。
"""

import argparse

import gradio as gr
import openai

# ============================================
# 配置
# ============================================
SYSTEM_PROMPT = """你是中国科学技术大学（中科大/USTC）智能问答助手。
你的职责是准确、友好地回答关于中科大的各类问题，包括但不限于：
- 学校概况、历史沿革
- 院系设置、学科优势
- 招生政策、报考指南
- 校园生活、住宿餐饮
- 科研成果、学术平台
- 就业深造、校友发展

请基于你所掌握的知识回答，如果不确定某个信息，请如实告知。"""


def create_chat_fn(api_base: str, model_name: str):
    """创建聊天函数"""
    client = openai.OpenAI(base_url=api_base, api_key="not-needed")

    def chat(message: str, history: list[list[str]]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": message})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )

            partial_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    partial_response += delta
                    yield partial_response

        except openai.APIConnectionError:
            yield "❌ 无法连接到推理服务，请确认 vLLM 服务已启动。\n运行: bash deploy/serve.sh"
        except Exception as error:
            yield f"❌ 发生错误: {error}"

    return chat


def build_demo(api_base: str, model_name: str) -> gr.Blocks:
    """构建 Gradio 界面"""
    chat_fn = create_chat_fn(api_base, model_name)

    with gr.Blocks(
        title="中科大智能问答助手",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎓 中科大智能问答助手
            基于 **Qwen2.5-7B + LoRA 微调 + vLLM 部署** 的全链路 LLM 项目

            > 你可以询问关于中国科学技术大学的任何问题！
            """
        )

        chatbot = gr.ChatInterface(
            fn=chat_fn,
            examples=[
                "中科大是什么时候成立的？",
                "中科大少年班有什么特色？",
                "中科大有哪些优势学科？",
                "如何报考中科大的研究生？",
                "中科大在人工智能领域有哪些成果？",
                "中科大毕业生就业情况怎么样？",
            ],
            retry_btn="🔄 重新生成",
            undo_btn="↩️ 撤销",
            clear_btn="🗑️ 清空对话",
        )

        gr.Markdown(
            """
            ---
            **技术栈**: LLaMA-Factory + DeepSpeed ZeRO-2 + PEFT LoRA + AWQ 量化 + vLLM

            **项目地址**: [yiyicat-llm](https://github.com/yiyicat/yiyicat-llm)
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="中科大智能问答助手 - Web 界面")
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API 地址",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/ustc-qa-quantized-awq-int4",
        help="模型名称（与 vLLM 启动时一致）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio 服务端口",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="是否创建公网分享链接",
    )
    args = parser.parse_args()

    demo = build_demo(args.api_base, args.model)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
