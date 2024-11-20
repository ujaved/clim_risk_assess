import webvtt
from chatbot import NumQuestionsParams, EmotionAnalysis, ModeAnalysis
from openai import OpenAI
import json
from chatbot import Chatbot
from datetime import datetime
from store import DBClient
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from difflib import get_close_matches


class MetricName(Enum):
    SPEAKING_TIME_MINS = "speaking_time_(mins)"
    FRACTION_SPEAKING_TIME = "fraction_speaking_time"
    NUM_TURNS = "num_turns"
    NUM_QUESTIONS = "num_questions"
    MEAN_WORD_SPEAD = "mean_num_words_per_sec"
    MEAN_NUM_WORDS_PER_TURN = "mean_num_words_per_turn"


@dataclass
class SpeakerStats:
    num_words: int = 0
    speaking_time: int = 0
    num_turns: int = 0
    num_questions: int = 0
    mean_word_speed: float = 0.0
    mean_words_per_turn: float = 0.0

    def transform(self, tot_speaking_time: int) -> dict:
        fraction_speaking_time = float(
            f"{float(self.speaking_time) / tot_speaking_time:.2f}"
        )
        return {
            MetricName.SPEAKING_TIME_MINS.value: self.speaking_time / 60,
            MetricName.FRACTION_SPEAKING_TIME.value: fraction_speaking_time,
            MetricName.NUM_TURNS.value: self.num_turns,
            MetricName.NUM_QUESTIONS.value: self.num_questions,
            MetricName.MEAN_WORD_SPEAD.value: self.mean_word_speed,
            MetricName.MEAN_NUM_WORDS_PER_TURN.value: self.mean_words_per_turn,
        }


class RecordingProcessor:

    def __init__(
        self,
        id: str,
        ts: datetime,
        transcript: str,
        teacher: str,
        chatbot: Chatbot,
        db_client: DBClient,
        name_mapping: dict[str, str],
    ) -> None:
        self.id = id
        self.ts = ts
        self.transcript = transcript
        self.dialogue: list[tuple[str, str, int, int]] = []
        self.speaker_stats: dict[str, SpeakerStats] = {}
        self.chatbot = chatbot
        self.db_client = db_client
        self.class_duration_secs = 0
        self.teacher_silence_secs = 0
        self.student_silence_secs = 0
        self.teacher = teacher
        self.name_mapping = name_mapping

    @property
    def date(self) -> str:
        return self.ts.date().isoformat()

    def get_num_questions(self) -> dict[str, list]:
        prompt = f"Following is a transcript of zoom classroom session. For each speaker tell me the number of questions they ask. \n\n {self.dialogue}"
        messages = [{"role": "user", "content": prompt}]
        response = OpenAI().chat.completions.create(
            model=self.chatbot.model_id,
            messages=messages,
            functions=[
                {
                    "name": "get_num_questions_asked",
                    "description": "get number of questions asked by each speaker in the given transcript",
                    "parameters": NumQuestionsParams.schema(),
                }
            ],
        )
        arguments = json.loads(response.choices[0].message.function_call.arguments)
        return arguments

    def get_emotion_analysis(self, interval: int) -> EmotionAnalysis:
        prompt = f"Following is a transcript of a group discussion with timestamps. For every interval of {interval} minutes, perform emotion analysis, choosing emotion labels from the given list. \n\n {self.transcript}"
        messages = [{"role": "user", "content": prompt}]
        response = OpenAI().beta.chat.completions.parse(
            model=self.chatbot.model_id,
            messages=messages,
            response_format=EmotionAnalysis,
        )
        return response.choices[0].message.parsed

    def get_mode_analysis(self, interval: int) -> ModeAnalysis:
        prompt = f"Following is a transcript of a classroom with timestamps. For every interval of {interval} minutes, classify it as a mode, choosing mode labels from the given list. \n\n {self.transcript}"
        messages = [{"role": "user", "content": prompt}]
        response = OpenAI().beta.chat.completions.parse(
            model=self.chatbot.model_id,
            messages=messages,
            response_format=ModeAnalysis,
        )
        return response.choices[0].message.parsed

    def process(self):
        prev_speaker = None
        prev_content = None
        prev_caption = None
        cur_turn_start = 0
        captions = webvtt.from_string(self.transcript)
        self.class_duration_secs = (
            captions[-1].end_in_seconds - captions[0].start_in_seconds
        )
        for caption in captions:
            fields = caption.text.split(":")
            if len(fields) < 2:
                continue
            speaker = fields[0].lower()
            if speaker in self.name_mapping:
                speaker = self.name_mapping[fields[0].lower()]
            if speaker not in self.speaker_stats:
                self.speaker_stats[speaker] = SpeakerStats()

            if prev_caption:
                silence = caption.start_in_seconds - prev_caption.end_in_seconds
                if speaker == self.teacher:
                    self.teacher_silence_secs += silence
                else:
                    self.student_silence_secs += silence

            content = fields[1]
            self.speaker_stats[speaker].num_words += len(content.strip().split())
            self.speaker_stats[speaker].speaking_time += (
                caption.end_in_seconds - caption.start_in_seconds
            )
            if prev_speaker == speaker:
                prev_content = f"{prev_content}{content}".strip()
            else:
                self.speaker_stats[speaker].num_turns += 1
                self.dialogue.append(
                    (
                        prev_speaker,
                        prev_content,
                        cur_turn_start,
                        caption.end_in_seconds,
                    )
                )
                prev_content = content
                cur_turn_start = caption.start_in_seconds
            prev_speaker = speaker
            prev_caption = caption

        self.dialogue = self.dialogue[1:]
        self.dialogue.append(
            (prev_speaker, prev_content, cur_turn_start, caption.end_in_seconds)
        )
        for stats in self.speaker_stats.values():
            stats.mean_word_speed = (
                (stats.num_words / stats.speaking_time)
                if stats.speaking_time > 0
                else 0.0
            )
            stats.mean_words_per_turn = (
                (stats.num_words / stats.num_turns) if stats.num_turns > 0 else 0.0
            )

        num_questions_json = self.db_client.get_num_questions(self.id)
        if not num_questions_json:
            num_questions_json = self.get_num_questions()
            self.db_client.insert_num_questions(self.id, num_questions_json)
        num_questions = NumQuestionsParams(**num_questions_json)
        for qc in num_questions.num_questions:
            # s = get_close_matches(qc.speaker, list(self.speaker_stats.values()))[0]
            s = self.name_mapping.get(qc.speaker) or qc.speaker
            self.speaker_stats[s].num_questions = qc.question_count


"""
chart_agg = (
            alt.Chart(df_agg)
            .mark_bar(color="green")
            .encode(
                alt.X("lead-follow", axis=alt.Axis(labelAngle=-45)),
                alt.Y("cond_prob").title(
                    "Conditional probability of (lead,follow) utterance"
                ),
                color=alt.Color("lead-follow"),
            )
            .properties(height=500)
            .add_params(alt.selection_point())
        )
"""
