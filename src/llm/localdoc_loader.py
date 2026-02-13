import os

from langchain_community.document_loaders import TextLoader

from llm.idocloader import IDocumentLoader


class LocalDocumentLoader(IDocumentLoader):
    def __init__(self, root_folder_src: str):
        self.root_folder_src = root_folder_src

    def _find_documents(self, path=None):
        root_path = path if path else self.root_folder_src
        files = []

        for root, _, filenames in os.walk(root_path):
            for filename in filenames:
                files.append({"name:": filename, "path": os.path.join(root, filename)})

        return files

    def _get_document_entities(self, document_path: str):
        loader = TextLoader(document_path)
        return loader.load()

    def load_document_entities(self, path=None):
        docs = self._find_documents(path)
        entities = []
        for i, doc in enumerate(docs):
            doc_entities = self._get_document_entities(doc["path"])
            entities.extend(doc_entities)
        return entities
