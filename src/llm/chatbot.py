import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from llm.prompt import initial_prompt
from llm.retriever import SimilarityRetriever

load_dotenv()


class ChatBotChain:
    def __init__(self, context_root_folder):
        api_key = os.getenv("GROK_API_KEY")

        self.chatbot = ChatGroq(
            model="openai/gpt-oss-20b",  # O el modelo específico que quieras usar (ej: grok-2)
            api_key=api_key,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    initial_prompt,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        self.retriever = SimilarityRetriever(context_root_folder)
        self.rag_chain_manual = (
            {
                "context": lambda x: self.retriever.as_retriever().invoke(
                    x["question"]
                ),
                "tdate": lambda x: datetime.now().strftime("%Y-%m-%d"),
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
            }
            | prompt
            | self.chatbot
            | StrOutputParser()
        )

    def interaction(self, input_text: str, messages_history: list):
        result = self.rag_chain_manual.invoke(
            {"history": messages_history, "question": input_text}
        )
        return result
