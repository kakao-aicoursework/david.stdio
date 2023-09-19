"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base


# openai.api_key = "<YOUR_OPENAI_API_KEY>"
openai.api_key = open(".key", "r").read().replace("\n","")


# parallel_example = {
#     "한국어": ["오늘 날씨 어때", "딥러닝 기반의 AI기술이 인기를끌고 있다."],
#     "영어": ["How is the weather today", "Deep learning-based AI technology is gaining popularity."],
#     "일본어": ["今日の天気はどうですか", "ディープラーニングベースのAIテクノロジーが人気を集めています。"]
# }


# def translate_text_using_text_davinci(text, src_lang, trg_lang) -> str:
#     response = openai.Completion.create(engine="text-davinci-003",
#                                         prompt=f"Translate the following {src_lang} text to {trg_lang}: {text}",
#                                         max_tokens=200,
#                                         n=1,
#                                         temperature=1
#                                         )
#     translated_text = response.choices[0].text.strip()
#     return translated_text


def translate_text_using_chatgpt(text) -> str:#, src_lang, trg_lang) -> str:
    # fewshot 예제를 만들고
    # def build_fewshot(src_lang, trg_lang):
    #     src_examples = parallel_example[src_lang]
    #     trg_examples = parallel_example[trg_lang]

    #     fewshot_messages = []

    #     for src_text, trg_text in zip(src_examples, trg_examples):
    #         fewshot_messages.append({"role": "user", "content": src_text})
    #         fewshot_messages.append({"role": "assistant", "content": trg_text})

    #     return fewshot_messages

    # system instruction 만들고
    # system_instruction = f"assistant는 번역앱으로서 동작한다. {src_lang}를 {trg_lang}로 적절하게 번역하고 번역된 텍스트만 출력한다."

    # messages를만들고
    # fewshot_messages = build_fewshot(src_lang=src_lang, trg_lang=trg_lang)

    messages = [
        # {"role": "system", "content": system_instruction},
                # *fewshot_messages,
                {"role": "user", "content": text}
                ]

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
    print(response)
    translated_text = response['choices'][0]['message']['content']
    # Return
    return translated_text


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
        self.messages = [
            Message(
                speaker="my",
                original_text=self.text,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                # to_lang=self.trg_lang,
            )
        ] + self.messages

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Translations will appear here."
        translated = translate_text_using_chatgpt(
            self.text)#, src_lang=self.src_lang, trg_lang=self.trg_lang)
        
        print(translated)

        self.messages = [
            Message(
                speaker="bot",
                original_text=translated,
                created_at=datetime.now().strftime("%Y %d %I:%M %p"),
                # to_lang=self.trg_lang,
            )
        ] + self.messages
        
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


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        # pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


# project_data = open("project_data_카카오싱크.txt", "r").read().replace("\n","")

def index():
    """The main view."""
    return pc.container(
        header(),
        pc.hstack(
            pc.vstack(
                pc.foreach(
                    State.messages,
                    lambda m: pc.text(m.created_at, "  [", m.speaker, "] ", m.original_text),
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
