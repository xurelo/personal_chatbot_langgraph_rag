from datetime import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm.chatbot import ChatBotChain

load_dotenv()


class ChatbotState(TypedDict):
    context: str
    question: str
    history: Annotated[list, add_messages]
    last_answer: str
    tdate: str


class ChatBotGraph:
    def _build_graph(self):
        workflow = StateGraph(ChatbotState)
        workflow.add_node("CHATBOT", self._chatbot_node)
        workflow.add_edge(START, "CHATBOT")
        workflow.add_edge("CHATBOT", END)

        return workflow.compile(checkpointer=self.checkpoint_memory)

    def __init__(self, chatbot: ChatBotChain):
        self.checkpoint_memory = MemorySaver()
        self.chatbot = chatbot
        self.graph = self._build_graph()

    def _chatbot_node(self, state):
        question = state["history"][-1].content
        answer = self.chatbot.interaction(question, state["history"][:-1])
        print(f"answer from chain:{answer}")
        new_state = {}
        new_state["last_answer"] = answer
        new_state["history"] = AIMessage(content=answer)
        return new_state

    def interaction(self, session_id: str, input_text: str):
        initial_state = {
            "tdate": datetime.now().strftime("%Y-%m-%d"),
            "history": [HumanMessage(content=input_text)],
        }
        config = {"configurable": {"thread_id": session_id}}
        output = self.graph.invoke(initial_state, config=config)
        return output["history"][-1].content
