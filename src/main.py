import os

import gradio as gr

from llm.audio import AudioTranscriber
from llm.chatbot import ChatBotChain
from llm.chatbot_graph import ChatBotGraph
from llm.gdrivedoc_loader import GDriveDocumentLoader
from llm.localdoc_loader import LocalDocumentLoader
from llm.retriever import SimilarityRetriever

path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "../documents/"

docloader = LocalDocumentLoader(path)
gdriveloader = GDriveDocumentLoader()


retriever = SimilarityRetriever(gdriveloader)
chatbot_chain = ChatBotChain(retriever)
llm_chatbot = ChatBotGraph(chatbot_chain)
transcriber = AudioTranscriber()


def interaction(chat, message: str, request: gr.Request):
    session_id = request.session_hash if request else "default"
    print(f"message is:{message}")
    chat.append({"role": "user", "content": message})
    response = llm_chatbot.interaction(session_id, message)
    print(f"llm response is:{response}")
    chat.append({"role": "assistant", "content": response})
    print(f"now chat of messages is:{chat}")
    return chat, ""


def process_voice(audio_path, chat, request: gr.Request):
    print(f"Process voice:{audio_path}")
    text = transcriber.transcribe(audio_path)
    print(f"Audio transcribed is:{text}")
    chat, _ = interaction(chat, text["text"], request)
    return chat, None


with gr.Blocks() as demo:
    ch = gr.Chatbot(
        value=[
            {
                "role": "assistant",
                "content": "¡Hola! 👋 Soy el asistente virtual de Fran. ¿En qué puedo ayudarte hoy?",
            }
        ],
        label="Francisco Manuel Romero CV Assistant",
    )
    with gr.Row():
        inp = gr.Textbox(label="User", placeholder="Ask something")
        audio_input = gr.Audio(
            sources=["microphone"], type="filepath", label="Press to talk"
        )

        audio_input.stop_recording(
            fn=process_voice, inputs=[audio_input, ch], outputs=[ch, audio_input]
        )
        inp.submit(fn=interaction, inputs=[ch, inp], outputs=[ch, inp])
    demo.launch(server_name="127.0.0.1", server_port=8081, share=True)
