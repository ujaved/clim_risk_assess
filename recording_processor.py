import webvtt
from typing import TypedDict
import pandas as pd
from chatbot import NumQuestionsParams
from openai import OpenAI
import json
from chatbot import Chatbot
from datetime import datetime


class SpeakerStats(TypedDict):
    num_words: int
    speaking_time: int
    num_turns: int
    num_questions: int
    mean_word_speed: float
    mean_words_per_turn: float

    @staticmethod
    def keys():
        return SpeakerStats.__dict__["__annotations__"].keys()


class ClassroomStats(TypedDict):
    num_questions: int


class RecordingProcessor:

    def __init__(
        self, name: str, ts: datetime, transcript: str, chatbot: Chatbot
    ) -> None:
        self.name = name
        self.ts = ts
        self.transcript = transcript
        self.dialogue: list[tuple[str, str]] = []
        self.speaker_stats: dict[str, SpeakerStats] = {}
        self.chatbot = chatbot

    def get_num_questions(self) -> None:
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
        for qc in NumQuestionsParams(**arguments).num_questions:
            self.speaker_stats[qc.speaker]["num_questions"] = qc.question_count

    def process(self):
        prev_speaker = None
        prev_content = None
        for caption in webvtt.from_string(self.transcript):
            fields = caption.text.split(":")
            speaker = fields[0].lower()
            if speaker not in self.speaker_stats:
                self.speaker_stats[speaker] = {
                    "num_words": 0,
                    "num_turns": 0,
                    "speaking_time": 0,
                    "mean_word_speed": 0.0,
                    "mean_words_per_turn": 0.0,
                }
            content = fields[1]
            self.speaker_stats[speaker]["num_words"] += len(content.strip().split())
            self.speaker_stats[speaker]["speaking_time"] += (
                caption.end_in_seconds - caption.start_in_seconds
            )
            if prev_speaker == speaker:
                prev_content = f"{prev_content}{content}".strip()
            else:
                self.speaker_stats[speaker]["num_turns"] += 1
                self.dialogue.append((prev_speaker, prev_content))
                prev_content = content
            prev_speaker = speaker

        self.dialogue = self.dialogue[1:]
        self.dialogue.append((prev_speaker, prev_content))
        for stats in self.speaker_stats.values():
            stats["mean_word_speed"] = stats["num_words"] / stats["speaking_time"]
            stats["mean_words_per_turn"] = stats["num_words"] / stats["num_turns"]
        self.get_num_questions()
