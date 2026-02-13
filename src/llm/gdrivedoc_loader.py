import base64
import json
import os
import tempfile

from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain_community.document_loaders.googledrive import GoogleDriveLoader

from llm.idocloader import IDocumentLoader

load_dotenv()


class GDriveDocumentLoader(IDocumentLoader):
    def __init__(self):
        cred = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        decoded_cred = base64.b64decode(cred).decode("utf-8")
        self.creds_dict = json.loads(decoded_cred)
        self.credentials = service_account.Credentials.from_service_account_info(
            self.creds_dict, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )

    def _get_document_entities(self, folder_id: str):
        """Load documents from a specific folder in Google Drive."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as temp_file:
            # Write JSON to a temporal file
            json.dump(self.creds_dict, temp_file)
            temp_file.flush()
            loader = GoogleDriveLoader(
                folder_id=folder_id,
                recursive=True,
                service_account_key=temp_file.name,
                file_types=["document"],
            )
            docs = loader.load()
        print(f"{len(docs)} documents from Drive have been loaded")
        print(f"docs are:{docs}")
        return docs

    def load_document_entities(self, path=None):
        folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        entities = self._get_document_entities(folder_id)
        return entities
