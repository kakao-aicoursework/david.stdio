from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

import os

os.environ["OPENAI_API_KEY"] = open(".key", "r").read().replace("\n","")

PWD=os.getcwd() # HOME_PATH
CHROMA_PERSIST_DIR = os.path.join(PWD, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


def upload_embedding_from_file(loader):
    documents = TextLoader(loader).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')

def load():
    files = ["project_data_카카오소셜.txt", "project_data_카카오싱크.txt","project_data_카카오톡채널.txt"]
    for file in files:
        file_path =  os.path.join(PWD, file)
        try:
            upload_embedding_from_file(file_path)
            print("SUCCESS: ", file_path)
        except Exception as e:
            print("FAILED: ", file_path + f" by({e})")    

if __name__ == "__main__":
    load()