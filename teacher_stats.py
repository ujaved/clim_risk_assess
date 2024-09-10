from dataclasses import dataclass, field
from recording_processor import RecordingProcessor
import streamlit as st
import pandas as pd


@st.cache_data
def get_stats_df(meeting_name: str):
    return st.session_state.meeting_to_rp[meeting_name].get_stats_df()


@dataclass
class TeacherStats:
    name: str
    recording_processors: list[RecordingProcessor]  # sorted by time
    students: set[str] = field(default_factory=set)
    tab: any = None

    def get_participation_stats_df(self) -> pd.DataFrame:
        data = []
        for rp in self.recording_processors:
            for speaker, stats in rp.speaker_stats.items():
                for stat_name in stats.keys():
                    data.append((rp.ts, speaker, stat_name, stats[stat_name]))
        df = pd.DataFrame(data, columns=["date", "speaker", "metric", "value"])
        return df

    def get_teacher_interruption_df(self) -> pd.DataFrame:
        data = []
        for rp in self.recording_processors:
            interruption_count = {}
            for i, d in enumerate(rp.dialogue[1 : len(rp.dialogue) - 1], start=1):
                # we consider 3-dialogue blocks with the middle (current) potentially representing a teacher interruption
                if d[0] != self.name or rp.dialogue[i - 1][0] != rp.dialogue[i + 1][0]:
                    continue
                student = rp.dialogue[i - 1][0]
                data.append(
                    {
                        "student": student,
                        "date": rp.ts,
                        "num_seconds_before_interruption": rp.dialogue[i - 1][2],
                    }
                )
        return pd.DataFrame(data)

    def get_pairwise_following_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = []
        pairwise_agg_count = {}
        for rp in self.recording_processors:
            dialogue_without_teacher = [d for d in rp.dialogue if d[0] != self.name]
            pairwise_count = {}
            for i, d in enumerate(dialogue_without_teacher[1:], start=1):
                if dialogue_without_teacher[i - 1][0] == d[0]:
                    continue
                pair = (dialogue_without_teacher[i - 1][0], d[0])

                count = pairwise_count[pair] if pair in pairwise_count else 0
                pairwise_count[pair] = count + 1

                count = pairwise_agg_count[pair] if pair in pairwise_agg_count else 0
                pairwise_agg_count[pair] = count + 1

            for p, count in pairwise_count.items():
                data.append(
                    {"lead": p[0], "follow": p[1], "count": count, "date": rp.ts}
                )
        return pd.DataFrame(data), pd.DataFrame(
            [
                {"lead": p[0], "follow": p[1], "count": count}
                for p, count in pairwise_agg_count.items()
            ]
        )

    def render(self):
        with self.tab:
            for rp in self.recording_processors:
                st.header(rp.name, divider=True)
                st.dataframe(get_stats_df(rp.name), use_container_width=True)
