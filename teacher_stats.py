from dataclasses import dataclass, field
from recording_processor import RecordingProcessor
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def get_stats_df(meeting_name: str):
    return st.session_state.meeting_to_rp[meeting_name].get_stats_df()


@dataclass
class TeacherStats:
    name: str
    recording_processors: list[RecordingProcessor]  # sorted by time
    students: set[str] = field(default_factory=set)
    tab: any = None

    def get_stats_df(self) -> pd.DataFrame:
        data = []
        for rp in self.recording_processors:
            for speaker, stats in rp.speaker_stats.items():
                for stat_name in stats.keys():
                    data.append((rp.ts, speaker, stat_name, stats[stat_name]))
        df = pd.DataFrame(data, columns=["date", "speaker", "metric", "value"])
        return df

    def render(self):
        with self.tab:
            for rp in self.recording_processors:
                st.header(rp.name, divider=True)
                st.dataframe(get_stats_df(rp.name), use_container_width=True)
