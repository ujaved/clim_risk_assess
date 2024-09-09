import streamlit as st
from utils import ZoomClient
import requests
from recording_processor import RecordingProcessor, SpeakerStats
from teacher_stats import TeacherStats
from chatbot import OpenAIChatbot
from dateutil import parser
import altair as alt
from supabase import create_client, Client
from gotrue.errors import AuthApiError
import pandas as pd
from altair import datum

WELCOME_MSG = "Welcome to Scientifically Taught Science"

# client = ZoomClient()


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


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


def add_recording_cb():

    if (
        not st.session_state.get("recording_url")
        or not st.session_state.get("recording_date")
        or not st.session_state.get("recording_class")
    ):
        return
    st.session_state.conn.table("recordings").insert(
        {
            "user_id": st.session_state.user.id,
            "link": st.session_state.recording_url,
            "date": st.session_state.recording_date.isoformat(),
            "class_id": st.session_state["class_name_id_mapping"][
                st.session_state.get("recording_class")
            ],
        }
    ).execute()


def new_class_cb():
    st.session_state.conn.table("classes").insert(
        {
            "name": st.session_state.new_class,
            "teacher_id": st.session_state.user.id,
        }
    ).execute()
    st.session_state.new_class = ""


def add_recording():

    classes = (
        st.session_state.conn.table("classes")
        .select("*")
        .eq("teacher_id", st.session_state["user"].id)
        .execute()
    )
    st.session_state["class_name_id_mapping"] = {
        cl["name"]: cl["id"] for cl in classes.data
    }

    with st.form("Add Recording", clear_on_submit=True):
        st.text_input("Zoom Recording URL", key="recording_url")
        st.selectbox(
            "Class", [cl["name"] for cl in classes.data], key="recording_class"
        )
        st.date_input("Recording date", value=None, key="recording_date")
        st.form_submit_button("Add", on_click=add_recording_cb)

    st.text_input(
        "Don't see the class name in the dropdown? Add a class",
        on_change=new_class_cb,
        key="new_class",
    )


def render_chart(rps: list[RecordingProcessor]):
    with st.container(border=True):
        name = (
            st.session_state.user.user_metadata["first_name"]
            + " "
            + st.session_state.user.user_metadata["last_name"]
        ).lower()
        teacher_stats = TeacherStats(name=name, recording_processors=rps)
        df = teacher_stats.get_stats_df()
        chart = (
            alt.Chart(df, title="Speaker stats")
            .mark_line(point=True, size=2)
            .encode(
                alt.X("date"),
                alt.Y("value").title(""),
                alt.Color("speaker"),
                tooltip=["value", "date"],
            )
            .properties(width=200)
            .facet(facet="metric", spacing=100, title="", columns=3)
            .resolve_scale(y="independent", x="independent")
        )
        show_teacher = st.checkbox("Show teacher data")
        if not show_teacher:
            chart = chart.transform_filter(datum.speaker != name)
        st.altair_chart(chart)


def dashboard():

    classes = (
        st.session_state.conn.table("classes")
        .select("*")
        .eq("teacher_id", st.session_state["user"].id)
        .execute()
    )
    st.session_state["class_name_id_mapping"] = {
        cl["name"]: cl["id"] for cl in classes.data
    }

    cl = st.sidebar.radio(
        "Classes Taught", st.session_state.class_name_id_mapping.keys()
    )

    recordings = (
        st.session_state["conn"]
        .table("recordings")
        .select("*")
        .eq("user_id", st.session_state["user"].id)
        .eq("class_id", st.session_state.class_name_id_mapping[cl])
        .execute()
    )
    if not recordings.data:
        return
    df = pd.DataFrame(recordings.data)
    df["has_transcript"] = [r["transcript"] is not None for r in recordings.data]
    # st.dataframe(df, column_order=["link", "date", "has_transcript"])

    rps = [
        RecordingProcessor(
            id=rec["id"],
            ts=parser.isoparse(rec["date"]),
            transcript=rec["transcript"],
            chatbot=OpenAIChatbot(model_id="gpt-4-turbo", temperature=0.0),
            db_client=st.session_state["conn"],
        )
        for _, rec in df.iterrows()
        if rec["has_transcript"]
    ]

    if not rps:
        return
    for rp in rps:
        rp.process()

    render_chart(rps)


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
        pg = st.navigation(
            [
                st.Page(
                    dashboard,
                    title="Dashboard",
                    icon=":material/dashboard:",
                    default=True,
                ),
                st.Page(add_recording, title="Add recording"),
            ]
        )
        pg.run()
    elif st.session_state.get("registered"):
        login()
    else:
        register_login()


#  teacher_tabs = []
#  if len(st.session_state.teacher_stats) > 0:
#      teacher_tab_names = [ts.name for ts in st.session_state.teacher_stats]
#      teacher_tab_names.reverse()
#      teacher_tabs = st.tabs(teacher_tab_names)
#  for i, t in enumerate(reversed(teacher_tabs)):
#      st.session_state.teacher_stats[i].tab = t
#     st.session_state.teacher_stats[i].render()


if __name__ == "__main__":
    main()
