from dataclasses import dataclass, field
from recording_processor import RecordingProcessor
import streamlit as st
import pandas as pd
from datetime import date


@st.cache_data
def get_stats_df(meeting_name: str):
    return st.session_state.meeting_to_rp[meeting_name].get_stats_df()


@dataclass
class TeacherStats:
    name: str
    recording_processors: list[RecordingProcessor]  # sorted by time
    students: set[str] = field(default_factory=set)
    tab: any = None

    def get_participation_stats_df(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        data = []
        for rp in self.recording_processors:
            if start_date and start_date > rp.ts.date():
                continue
            if end_date and end_date < rp.ts.date():
                continue
            for speaker, stats in rp.speaker_stats.items():
                transformed_stats = stats.transform(rp.class_duration_in_secs)
                for stat_name in transformed_stats.keys():
                    data.append(
                        (
                            rp.ts.date().isoformat(),
                            speaker,
                            stat_name,
                            transformed_stats[stat_name],
                        )
                    )
        df = pd.DataFrame(data, columns=["date", "speaker", "metric", "value"])
        return df

    def get_teacher_interruption_df(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        data = []
        for rp in self.recording_processors:
            if start_date and start_date > rp.ts.date():
                continue
            if end_date and end_date < rp.ts.date():
                continue
            for i, d in enumerate(rp.dialogue[1 : len(rp.dialogue) - 1], start=1):
                # we consider 3-dialogue blocks with the middle (current) potentially representing a teacher interruption
                if d[0] != self.name or rp.dialogue[i - 1][0] != rp.dialogue[i + 1][0]:
                    continue
                student = rp.dialogue[i - 1][0]
                data.append(
                    {
                        "student": student,
                        "date": rp.ts.date().isoformat(),
                        "num_seconds_before_interruption": rp.dialogue[i - 1][2],
                    }
                )
        return pd.DataFrame(data)

    def get_pairwise_following_df(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = []
        pairwise_agg_count = {}
        num_agg_turns = {}
        for rp in self.recording_processors:
            if start_date and start_date > rp.ts.date():
                continue
            if end_date and end_date < rp.ts.date():
                continue
            dialogue_without_teacher = [d for d in rp.dialogue if d[0] != self.name]
            pairwise_count = {}
            num_turns = {}
            for i, d in enumerate(dialogue_without_teacher[1:], start=1):
                if dialogue_without_teacher[i - 1][0] == d[0]:
                    continue

                pair = (dialogue_without_teacher[i - 1][0], d[0])

                num_turns[d[0]] = num_turns.get(d[0], 0) + 1
                num_agg_turns[d[0]] = num_agg_turns.get(d[0], 0) + 1
                pairwise_count[pair] = pairwise_count.get(pair, 0) + 1
                pairwise_agg_count[pair] = pairwise_agg_count.get(pair, 0) + 1

            for p, count in pairwise_count.items():
                normalized_count = float(f"{float(count) / num_turns[p[1]]:.2f}")
                data.append(
                    {
                        "lead-follow": f"{p[0]}-{p[1]}",
                        "num_turns_follow": num_turns[p[1]],
                        "normalized_count": normalized_count,
                        "date": rp.ts,
                    }
                )
        return pd.DataFrame(data), pd.DataFrame(
            [
                {
                    "lead-follow": f"{p[0]}-{p[1]}",
                    "num_turns_follow": num_agg_turns[p[1]],
                    "normalized_count": float(
                        f"{float(count) / num_agg_turns[p[1]]:.2f}"
                    ),
                }
                for p, count in pairwise_agg_count.items()
            ]
        )

    def render(self):
        with self.tab:
            for rp in self.recording_processors:
                st.header(rp.name, divider=True)
                st.dataframe(get_stats_df(rp.name), use_container_width=True)
