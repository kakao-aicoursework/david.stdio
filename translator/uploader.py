from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

import os

os.environ["OPENAI_API_KEY"] = open(".key", "r").read().replace("\n","")

PWD=os.getcwd() # HOME_PATH
CHROMA_PERSIST_DIR = os.path.join(PWD, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


def upload_embedding_from_file(loader, label):
    #add labels for each document
    documents = TextLoader(loader).load()

    text_splitter = CharacterTextSplitter(separator="\n#", chunk_size=500, chunk_overlap=100) # devide by meanings
    docs = text_splitter.split_documents(documents)
    for doc in docs:
        doc.page_content = f"Title: {label}\n{doc.page_content}"

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
        label = file.replace(".txt","").split("_")[2]

        try:
            upload_embedding_from_file(file_path, label)
            print("SUCCESS: ", file_path)
        except Exception as e:
            print("FAILED: ", file_path + f" by({e})")    

if __name__ == "__main__":
    load()