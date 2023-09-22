"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime
import tiktoken

import pynecone as pc
from pynecone.base import Base
from langchain.llms import OpenAI

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

import translator.uploader as uld


# openai.api_key = "<YOUR_OPENAI_API_KEY>"
import os


PWD = os.getcwd()
HISTORY_DIR = os.path.join(PWD, "./chat_histories/")
CHROMA_PERSIST_DIR = os.path.join(PWD, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"

os.environ["OPENAI_API_KEY"] = open(".key", "r").read().replace("\n","")



def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

def truncate_text(text, max_tokens=3000):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:  # 토큰 수가 이미 3000 이하라면 전체 텍스트 반환
        return text
    return enc.decode(tokens[:max_tokens])


class Message(Base):
    speaker: str=""
    original_text: str
    created_at: str
    # to_lang: str


class State(pc.State):
    """The app state."""

    speaker: str=""
    speakers: list[str] = []
    text: str = ""
    messages: list[Message] = []
    # src_lang: str = "한국어"
    # trg_lang: str = "영어"
    # project_data: str=""

    # def __init__(self):
    #     self.project_data = open("project_data_카카오싱크.txt", "r").read().replace("\n","")

    def set_speaker(self, speakers: list[str]):
        self.speakers = speakers
    
    def speaker_change(self, speaker: str):
        self.speaker = speaker

    def post(self):
        self.messages = self.messages + [
            Message(
                speaker="👨🏻‍💻",
                original_text=self.text,
                created_at=datetime.now().strftime("%Y %d %I:%M %p\n"),
                # to_lang=self.trg_lang,
            )
        ]
        self.output()

    # @pc.var
    def output(self) -> str:
        if not self.text:
            return "no text"
        if not self.text.strip():
            return "Translations will appear here."
        translated = gernerate_answer(self.text)#translate_text_using_chatgpt(self.text)
        self.messages = self.messages + [
            Message(
                speaker="🤖",
                original_text=translated,
                created_at=datetime.now().strftime("%Y %d %I:%M %p\n"),
            )
        ] 
        
        # return translated


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("GPT Prompt 🗺", font_size="2rem"),
        pc.text(
            "Ask to GPT!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            pc.box(
                # pc.text(message.to_lang),
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


# def output():
#     return pc.box(
#         pc.box(
#             smallcaps(
#                 "Output",
#                 color="#aeaeaf",
#                 background_color="white",
#                 padding_x="0.1rem",
#             ),
#             position="absolute",
#             top="-0.5rem",
#         ),
#         # pc.text(State.output),
#         padding="1rem",
#         border="1px solid #eaeaef",
#         margin_top="1rem",
#         border_radius="8px",
#         position="relative",
#     )

#####


def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer

#####

_db = Chroma(
    persist_directory= CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)

_retriever = _db.as_retriever()

def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

def gernerate_answer(user_message, conversation_id: str='mybot') -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    
    prompt=os.path.join(PWD, "prompt")
    parse_intent_chain = LLMChain(llm=llm, prompt=
            ChatPromptTemplate.from_template(
            template=truncate_text(read_prompt_template(prompt))
        ))


    history_file = load_conversation_history(conversation_id)
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]

    context["related_documents"] = query_db(context["user_message"])
    context["chat_history"] = get_chat_history(conversation_id)

    # intent = parse_intent_chain(context)["intent"]
    answer = parse_intent_chain.run(context)


    # if intent == "bug":
    #     context["related_documents"] = query_db(context["user_message"])

    #     answer = ""
    #     for step in [bug_step1_chain, bug_step2_chain]:
    #         context = step(context)
    #         answer += context[step.output_key]
    #         answer += "\n\n"
    # elif intent == "enhancement":
    #     answer = enhance_step1_chain.run(context)
    # else:
    #     context["related_documents"] = query_db(context["user_message"])
    #     context["compressed_web_search_results"] = query_web_search(
    #         context["user_message"]
    #     )
    #     answer = default_chain.run(context)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return answer
        
def message(message: Message):
    return pc.box(
        pc.vstack(
            pc.text(message.created_at, " ", message.speaker),
            pc.text_area(value=message.original_text),
            spacing="1rem",
            position="relative",
            align_items="left"
            )
    )

def index():
    if os.path.exists(f"{HISTORY_DIR}/mybot.json"):
        os.remove(f"{HISTORY_DIR}/mybot.json")
    
    if not os.path.exists(CHROMA_PERSIST_DIR):
        uld.load()

    """The main view."""
    return pc.container(
        header(),
        pc.hstack(
            pc.vstack(
                pc.foreach(
                    State.messages,
                    lambda m: message(m),
                ),
                width="100rem",
                margin_right="1rem",
                align_items="left"
            ),
            pc.vstack(
                pc.input(
                        placeholder="Ask to GPT",
                        on_blur=State.set_text,
                        margin_top="1rem",
                        border_color="#eaeaef",
                        width="20rem",
                        align_items="left"
                    ),
                    pc.button("Post", on_click=State.post, margin_top="1rem")
                ),
        ),
        # width="100rem",
        aligh_items="left",
        padding="2rem",
        max_width="1000px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="GPT Prompt")
app.compile()
