import streamlit as st
from utils import ZoomClient
import requests
from recording_processor import RecordingProcessor, SpeakerStats
from teacher_stats import TeacherStats
from chatbot import OpenAIChatbot
from dateutil import parser
import altair as alt
from supabase import create_client
from gotrue.errors import AuthApiError
import pandas as pd


WELCOME_MSG = "Welcome to Scientifically Taught Science"

# client = ZoomClient()


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def teachers_selection_cb():
    if not st.session_state.teachers_selected:
        return
    teachers_selected = [t.lower() for t in st.session_state.teachers_selected]
    for teacher in teachers_selected:

        with st.container(border=True):
            teacher_stats = TeacherStats(
                name=teacher,
                recording_processors=[
                    rp for rp in st.session_state.rps if teacher in rp.speaker_stats
                ],
            )
            df = teacher_stats.get_stats_df()
            st.altair_chart(
                alt.Chart(df, title="Student stats")
                .mark_line(
                    point=True,
                    size=2,
                )
                .encode(
                    alt.X("date"),
                    alt.Y("value").title(""),
                    alt.Color("student"),
                    tooltip=["value", "date"],
                )
                .properties(width=200)
                .facet(facet="metric", spacing=100, title="", columns=3)
                .resolve_scale(y="independent", x="independent"),
            )


def initialize_state(access_tokens: dict[str, str]):
    if st.session_state.get("disable_teacher_selection") is None:
        st.session_state["disable_teacher_selection"] = False
    if st.session_state.get("teacher_stats") is None:
        st.session_state.teacher_stats = []

    if "rps" in st.session_state:
        return
    st.session_state.rps = []
    with st.spinner("retrieving zoom meetings"):
        recs = client.get_recordings(num_lookback_months=6)

    for meeting in recs:
        transcript_download_url = [
            f["download_url"]
            for f in meeting["recording_files"]
            if f["file_type"] == "TRANSCRIPT"
        ][0]
        transcript_download_url = (
            f"{transcript_download_url}?access_token={client.access_token}"
        )
        name = f"{meeting['topic']}_{meeting['start_time']}"
        rp = RecordingProcessor(
            name=name,
            transcript=requests.get(transcript_download_url).text,
            ts=parser.isoparse(meeting["start_time"]),
            chatbot=OpenAIChatbot(model_id="gpt-4-turbo", temperature=0.0),
        )
        rp.process()
        st.session_state.rps.append(rp)
    st.session_state.rps = sorted(st.session_state.rps, key=lambda r: r.ts)


def add_recording():
    if not st.session_state.get("recording_url") or not st.session_state.get(
        "recording_date"
    ):
        return
    st.session_state.conn.table("recordings").insert(
        {
            "user_id": st.session_state.user.id,
            "link": st.session_state.recording_url,
            "date": st.session_state.recording_date.isoformat(),
        }
    ).execute()


def account():

    with st.form("Add Recording", clear_on_submit=True):
        st.text_input("Add a zoom recording share URL", key="recording_url")
        st.date_input("Recording date", value=None, key="recording_date")
        st.form_submit_button("Add", on_click=add_recording)

    recordings = (
        st.session_state["conn"]
        .table("recordings")
        .select("*")
        .eq("user_id", st.session_state["user"].id)
        .execute()
    )
    if not recordings.data:
        return
    df = pd.DataFrame(recordings.data)
    df["has_transcript"] = [r["transcript"] is not None for r in recordings.data]
    st.dataframe(df, column_order=["link", "date", "has_transcript"])

    rps = [
        RecordingProcessor(
            name=rec["date"],
            ts=parser.isoparse(rec["date"]),
            transcript=rec["transcript"],
            chatbot=OpenAIChatbot(model_id="gpt-4-turbo", temperature=0.0),
        )
        for _, rec in df.iterrows()
        if rec["has_transcript"]
    ]

    if not rps:
        return
    for rp in rps:
        rp.process()

    with st.container(border=True):
        name = (
            st.session_state.user.user_metadata["first_name"]
            + " "
            + st.session_state.user.user_metadata["last_name"]
        ).lower()
        teacher_stats = TeacherStats(name=name, recording_processors=rps)
        df = teacher_stats.get_stats_df()
        st.altair_chart(
            alt.Chart(df, title="Student stats")
            .mark_line(
                point=True,
                size=2,
            )
            .encode(
                alt.X("date"),
                alt.Y("value").title(""),
                alt.Color("student"),
                tooltip=["value", "date"],
            )
            .properties(width=200)
            .facet(facet="metric", spacing=100, title="", columns=3)
            .resolve_scale(y="independent", x="independent"),
        )


def login_submit(is_login: bool):
    if is_login:
        try:
            data = st.session_state["conn"].auth.sign_in_with_password(
                {
                    "email": st.session_state.login_email,
                    "password": st.session_state.login_password,
                }
            )
            st.session_state["authenticated"] = True
            st.session_state["user"] = data.user
        except AuthApiError as e:
            st.error(e)
        return

    try:
        st.session_state["conn"].auth.sign_up(
            {
                "email": st.session_state.register_email,
                "password": st.session_state.register_password,
                "options": {
                    "data": {
                        "first_name": st.session_state.register_first_name,
                        "last_name": st.session_state.register_last_name,
                    },
                },
            }
        )
        st.session_state["registered"] = True
    except AuthApiError as e:
        st.error(e)


def login():
    with st.form("login_form", clear_on_submit=True):
        st.text_input("Email", key="login_email")
        st.text_input("Password", type="password", key="login_password")
        st.form_submit_button("Login", on_click=login_submit, args=[True])


def register_login():
    login_tab, register_tab = st.tabs(["Login", "Sign up"])
    with login_tab:
        with st.form("login_form", clear_on_submit=True):
            st.text_input("Email", key="login_email")
            st.text_input("Password", type="password", key="login_password")
            st.form_submit_button("Submit", on_click=login_submit, args=[True])

    with register_tab:
        with st.form("register", clear_on_submit=True):
            st.text_input("Email", key="register_email")
            st.text_input("Password", type="password", key="register_password")
            st.text_input("First name", key="register_first_name")
            st.text_input("Last name", key="register_last_name")
            st.form_submit_button("Submit", on_click=login_submit, args=[False])


def main():

    st.set_page_config(page_title="STS", page_icon=":teacher:", layout="wide")
    st.session_state["conn"] = init_connection()
    if st.session_state.get("authenticated"):
        account()
    elif st.session_state.get("registered"):
        login()
    else:
        register_login()

    """
    st.sidebar.multiselect(
        "Teachers",
        teacher2access_token.keys(),
        key="teachers_selected",
        on_change=teachers_selection_cb,
    )
    initialize_state(teacher2access_token)

    teacher_tabs = []
    if len(st.session_state.teacher_stats) > 0:
        teacher_tab_names = [ts.name for ts in st.session_state.teacher_stats]
        teacher_tab_names.reverse()
        teacher_tabs = st.tabs(teacher_tab_names)
    for i, t in enumerate(reversed(teacher_tabs)):
        st.session_state.teacher_stats[i].tab = t
        st.session_state.teacher_stats[i].render()
    """


if __name__ == "__main__":
    main()
