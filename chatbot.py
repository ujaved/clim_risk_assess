from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from pydantic import BaseModel, Field

PROMPT_TEMPLATE = """
Current conversation: {history}
Human: {input}
AI:
"""

class QuestionCount(BaseModel):
    speaker: str = Field(description="name of speaker")
    question_count: int = Field(description="number of questions asked by the speaker")

class NumQuestionsParams(BaseModel):
    """get number of questions asked by each speaker in the transcript"""
    num_questions: list[QuestionCount] = Field(description="list of question counts for each speaker")


class StreamlitStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.container = st.empty()

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

class Chatbot:
    def __init__(self, model_id: str, temperature: float) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.num_tokens = 0
        self.num_tokens_delta = 0

    def response(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatbot(Chatbot):

    def __init__(self, model_id: str, temperature: float) -> None:
        super().__init__(model_id=model_id, temperature=temperature)
        self.llm = ChatOpenAI(model_name=self.model_id,temperature=self.temperature, streaming=True)
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)
        self.chain = ConversationChain(prompt=PROMPT, llm=self.llm, memory=ConversationBufferMemory(), verbose=True)


    def response(self, prompt: str) -> str:
        with get_openai_callback() as cb:
            # for every response, we create a new stream handler; if not, response would use the old container
            self.chain.llm.callbacks = [StreamlitStreamHandler()]
            resp = self.chain.run(prompt)
            self.num_tokens_delta = cb.total_tokens - self.num_tokens
            self.num_tokens = cb.total_tokens
        return resp