from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage.in_memory import InMemoryStore
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from llm.docloader import DocumentLoader


class SimilarityRetriever:
    def __init__(self, root_folder_src: str):
        self.embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        loader = DocumentLoader(root_folder_src)
        docs = loader.load_document_entities()
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self.vectorstore = Chroma(
            persist_directory="./mi_chroma_db",
            embedding_function=self.embed,
            collection_name="mi_cv",
        )
        docstore = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        # ids = [f"doc_{i}" for i, _ in enumerate(docs)]
        self.retriever.search_kwargs = {"k": 3}
        print(f"Número de documentos a añadir: {len(docs)}")
        # self.retriever.add_documents(docs, ids=ids)
        self.retriever.add_documents(docs)

    def search(self, query: str):
        return self.retriever.invoke(query)

    def as_retriever(self):
        return self.retriever
