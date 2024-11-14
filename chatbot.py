from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict

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

    num_questions: list[QuestionCount] = Field(
        description="list of question counts for each speaker"
    )


class EmotionName(str, Enum):
    joyful = "joyful"
    interested = "interested"
    proud = "proud"
    accepted = "accepted"
    optimistic = "optimistic"
    intimate = "intimate"
    peaceful = "peaceful"
    powerful = "powerful"

    guilty = "guilty"
    abandoned = "abandoned"
    despair = "despair"
    depressed = "depressed"
    lonely = "lonely"
    bored = "bored"

    hurt = "hurt"
    threatened = "threatened"
    hateful = "hateful"
    mad = "mad"
    aggressive = "aggressive"
    frustrated = "frustrated"
    distant = "distant"
    critical = "critical"

    startled = "startled"
    confused = "confused"
    amazed = "amazed"
    excited = "excited"

    disapproval = "disapproval"
    disappointed = "disappointed"
    awful = "awful"
    avoidance = "avoidance"

    humiliated = "humiliated"
    rejected = "rejected"
    submissive = "submissive"
    insecure = "insecure"
    anxious = "anxious"
    scared = "scared"


SecondaryToPrimaryMapping: dict[EmotionName, str] = {
    EmotionName.joyful: "happy",
    EmotionName.interested: "happy",
    EmotionName.proud: "happy",
    EmotionName.accepted: "happy",
    EmotionName.optimistic: "happy",
    EmotionName.intimate: "happy",
    EmotionName.peaceful: "happy",
    EmotionName.powerful: "happy",
    EmotionName.guilty: "sad",
    EmotionName.abandoned: "sad",
    EmotionName.despair: "sad",
    EmotionName.depressed: "sad",
    EmotionName.lonely: "sad",
    EmotionName.bored: "sad",
    EmotionName.hurt: "anger",
    EmotionName.threatened: "anger",
    EmotionName.hateful: "anger",
    EmotionName.mad: "anger",
    EmotionName.aggressive: "anger",
    EmotionName.frustrated: "anger",
    EmotionName.distant: "anger",
    EmotionName.critical: "anger",
    EmotionName.startled: "surprise",
    EmotionName.confused: "surprise",
    EmotionName.amazed: "surprise",
    EmotionName.excited: "surprise",
    EmotionName.disapproval: "disgust",
    EmotionName.disappointed: "disgust",
    EmotionName.awful: "disgust",
    EmotionName.avoidance: "disgust",
    EmotionName.humiliated: "fear",
    EmotionName.rejected: "fear",
    EmotionName.submissive: "fear",
    EmotionName.insecure: "fear",
    EmotionName.anxious: "fear",
    EmotionName.scared: "fear",
}


class ModeName(str, Enum):
    lecture = "Lecture"
    discussion = "Discussion"
    problem_solving = "Problem-Solving"
    warm_up = "Warm-Up (Icebreaker)"
    q_a = "Q&A (Question and Answer)"
    wrap_up = "Wrap up"


PrimaryToSecondaryMapping: dict[str, list[str]] = defaultdict(list)
for k, v in SecondaryToPrimaryMapping.items():
    PrimaryToSecondaryMapping[v].append(k.value)


class Emotion(BaseModel):
    start_time: str = Field(description="Start time of the interval")
    end_time: str = Field(description="End time of the interval")
    labels: list[EmotionName] = Field(description="Emotion labels detected")
    reasoning: str = Field(description="short description of reasoning")


class Mode(BaseModel):
    start_time: str = Field(description="Start time of the interval")
    end_time: str = Field(description="End time of the interval")
    label: ModeName = Field(description="mode label detected")
    reasoning: str = Field(description="short description of reasoning")


class EmotionAnalysis(BaseModel):
    """Emotion analysis in the transcript"""

    emotions: list[Emotion] = Field(
        description="list of emotions detected every specified interval"
    )


class ModeAnalysis(BaseModel):
    """Mode analysis in the transcript of a classroom"""

    modes: list[Mode] = Field(description="mode detected every specified interval")


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
        self.llm = ChatOpenAI(
            model_name=self.model_id, temperature=self.temperature, streaming=True
        )
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=PROMPT_TEMPLATE
        )
        self.chain = ConversationChain(
            prompt=PROMPT, llm=self.llm, memory=ConversationBufferMemory(), verbose=True
        )

    def response(self, prompt: str) -> str:
        with get_openai_callback() as cb:
            # for every response, we create a new stream handler; if not, response would use the old container
            self.chain.llm.callbacks = [StreamlitStreamHandler()]
            resp = self.chain.run(prompt)
            self.num_tokens_delta = cb.total_tokens - self.num_tokens
            self.num_tokens = cb.total_tokens
        return resp
