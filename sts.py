import streamlit as st
from recording_processor import RecordingProcessor, MetricName
from teacher_stats import TeacherStats
from chatbot import OpenAIChatbot
from dateutil import parser
import altair as alt
from supabase import create_client, Client
from gotrue.errors import AuthApiError
import pandas as pd
from altair import datum, expr
from streamlit_option_menu import option_menu

WELCOME_MSG = "Welcome to Scientifically Taught Science"

# client = ZoomClient()


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def initialize_state():
    if "class_name_id_mapping" not in st.session_state:
        classes = (
            st.session_state.conn.table("classes")
            .select("*")
            .eq("teacher_id", st.session_state["user"].id)
            .execute()
        )
        st.session_state["class_name_id_mapping"] = {
            cl["name"]: cl["id"] for cl in classes.data
        }


def add_recording_cb():

    if (
        not st.session_state.get("recording_url")
        or not st.session_state.get("recording_date")
        or not st.session_state.get("recording_class")
    ):
        st.error("Input data is missing")
        return
    try:
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
        st.success("Recording added successfully")
    except Exception as e:
        st.error(str(e))


def new_class_cb():
    st.session_state.conn.table("classes").insert(
        {
            "name": st.session_state.new_class,
            "teacher_id": st.session_state.user.id,
        }
    ).execute()
    st.session_state.new_class = ""
    # remove mapping so it gets refreshed
    del st.session_state["class_name_id_mapping"]


def add_recording():
    with st.form("Add Recording", clear_on_submit=True):
        st.text_input("Zoom Recording URL", key="recording_url")
        st.selectbox(
            "Class",
            st.session_state.class_name_id_mapping.keys(),
            key="recording_class",
        )
        st.date_input("Recording date", value=None, key="recording_date")
        st.form_submit_button("Add", on_click=add_recording_cb)

    st.text_input(
        "Don't see the class name in the dropdown? Add a class",
        on_change=new_class_cb,
        key="new_class",
    )


def render_participation_charts(teacher_stats: TeacherStats):
    st.subheader("Aggregate Participation Metrics", divider=True)
    show_teacher_data = st.checkbox("Show teacher data")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        start_date = col1.date_input("start date", value=None, key="start_date")
        end_date = col2.date_input("start date", value=None, key="end_date")
        df = teacher_stats.get_participation_stats_df(start_date, end_date)
        df_silence = teacher_stats.get_silence_df(start_date, end_date)

        st.subheader("Speaking Time (minutes)")
        speaking_time_chart = (
            alt.Chart(df)
            .mark_line(point=True, size=2)
            .encode(
                alt.X("date:O"),
                alt.Y("value").title(None),
                alt.Color("speaker"),
                tooltip=["value", "date"],
            )
        )

        col3, col4 = st.columns(2)
        do_frac_class_duration = col3.checkbox("As fraction of class duration")
        add_silence = col4.checkbox("Add silence")
        silence_chart = None
        if do_frac_class_duration:
            speaking_time_chart = speaking_time_chart.transform_filter(
                datum.metric == MetricName.FRACTION_SPEAKING_TIME.value
            )
            if add_silence:
                silence_chart = (
                    alt.Chart(df_silence)
                    .mark_line(size=4, strokeDash=[8, 8], point=True)
                    .encode(
                        alt.X("date:O"),
                        alt.Y("silence_fraction"),
                        alt.Color("silence_type"),
                    )
                )
        else:
            speaking_time_chart = speaking_time_chart.transform_filter(
                datum.metric == MetricName.SPEAKING_TIME_MINS.value
            )
            if add_silence:
                silence_chart = (
                    alt.Chart(df_silence)
                    .mark_line(size=4, strokeDash=[8, 8], point=True)
                    .encode(
                        alt.X("date:O"),
                        alt.Y("silence_(mins)"),
                        alt.Color("silence_type"),
                    )
                )
        if not show_teacher_data:
            speaking_time_chart = speaking_time_chart.transform_filter(
                datum.speaker != teacher_stats.name
            )
        if silence_chart:
            speaking_time_chart = (speaking_time_chart + silence_chart).resolve_scale(
                color="independent"
            )
        st.altair_chart(speaking_time_chart, use_container_width=True)
        st.divider()

        chart = (
            alt.Chart(df)
            .mark_line(point=True, size=2)
            .encode(
                alt.X("date:O"),
                alt.Y("value").title(None),
                alt.Color("speaker"),
                tooltip=["value", "date"],
            )
            .properties(width=300)
            .facet(facet="metric", spacing=100, title="", columns=2)
            .resolve_scale(y="independent", x="independent")
            .transform_filter(
                (datum.metric != MetricName.SPEAKING_TIME_MINS.value)
                & (datum.metric != MetricName.FRACTION_SPEAKING_TIME.value)
            )
        )
        if not show_teacher_data:
            chart = chart.transform_filter(datum.speaker != teacher_stats.name)
        st.altair_chart(chart)


def render_interruption_charts(teacher_stats: TeacherStats):
    if st.sidebar.checkbox("**Training examples**", value=True):
        st.subheader("Label a teacher utterance as interruption or not", divider=True)
        date = st.selectbox(
            "Date", [rp.ts.date() for rp in teacher_stats.recording_processors]
        )
        idx = [rp.ts.date() for rp in teacher_stats.recording_processors].index(date)
        convs = teacher_stats.recording_processors[idx].get_2_way_conversations(
            teacher_stats.name
        )
        student = st.selectbox("Student", convs.keys(), index=None)
        if student:
            studnet_convs = convs[student]
            for i, conv in enumerate(studnet_convs):
                for j, c in enumerate(conv):
                    if c[0] == teacher_stats.name:
                        with st.chat_message("ai"):
                            col1, col2 = st.columns(2)
                            col1.write(c[1])
                            col2.feedback(key=student + str(i) + str(j))
                    else:
                        with st.chat_message("user"):
                            st.write(c[1])
                st.divider()
        return

    st.subheader("Possible interruptions by teacher", divider=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        df = teacher_stats.get_teacher_interruption_df(
            col1.date_input("start date", value=None),
            col2.date_input("end date", value=None),
        )
        chart = alt.Chart(df).mark_line(point=True, size=2)
        if st.checkbox("Mean duration in seconds before interruption", value=True):
            chart = chart.encode(
                x="date:O", y="mean(num_seconds_before_interruption)", color="student"
            )
        else:
            chart = chart.encode(
                x="date:O", y="count(num_seconds_before_interruption)", color="student"
            )
        st.altair_chart(chart, use_container_width=True)


def render_pairwise_charts(teacher_stats: TeacherStats):
    st.subheader(
        "Pairwise participation for a (lead, follow) pair, given the data, is the conditional probability that the follow speaks right after the lead, divided by random chance that the follow speaks. ",
        divider=True,
    )
    with st.container(border=True):
        col1, col2 = st.columns(2)
        df = teacher_stats.get_pairwise_following_df(
            col1.date_input("start date", value=None),
            col2.date_input("end date", value=None),
        )

        st.info(
            "Click on a cell to get a concrete description for the (lead,follow) participation metric"
        )

        # threshold = st.slider("threshold fraction", value=0.2, step=0.02)
        # filtered_df_agg = df_agg.loc[(df_agg["num_turns_follow"] > 10) & (df_agg["cond_prob"] >= threshold)]
        # df.loc[(df["cond_prob"] < threshold) | (df["num_turns_follow"] < 10),"cond_prob"] = 0.0

        matrix = (
            alt.Chart(df)
            .mark_rect(color="orange", size=10)
            .encode(
                alt.X("follow"),
                alt.Y("lead"),
                alt.Color("ratio")
                .scale(scheme="yellowgreen")
                .title("pairwise participation ratio"),
            )
            .properties(height=500)
            .add_params(alt.selection_point())
        )
        selection = st.altair_chart(
            matrix, use_container_width=True, on_select="rerun"
        ).selection.param_1
        if selection:
            lead = selection[0]["lead"]
            follow = selection[0]["follow"]
            row = df[(df["lead"] == lead) & (df["follow"] == follow)].iloc[0]
            p_lf = "P_{lf}"
            p_f_l = "P_{f|l}"
            explanation = f"""
            Excluding teacher utterances, the probability of follow **{follow}** speaking $P_f$ is {row['prob_turns_follow']:.2f}.
            The probability of lead **{lead}** speaking $P_l$ is {row['prob_turns_lead']:.2f}.
            The empirical probability of the pair **({lead}-{follow})** ${p_lf}$ is (number of pair occurences)/(number of total pair occurences) = {row['prob_pairwise']:.2f}.
            The conditional probability of **{follow}**'s turn after **{lead}** ${p_f_l}$ = ${p_lf}$/$P_l$ = {row['prob_pairwise']:.2f}/{row['prob_turns_lead']:.2f} = {row['prob_cond']:.2f}.
            The pairwise participation ratio is ${p_f_l}$ divided by the random chance of follow's turn, i.e., ${p_f_l}$/$P_f$ = {row['prob_cond']:.2f}/{row['prob_turns_follow']:.2f} = {row['ratio']:.2f}.
            """
            st.write(explanation)
            # f"Conditional probability of {follow}'s turn after {lead}'s turn = (number of {lead}-{follow} utterances) /  number of {lead} turns = {selection[0]['prob_cond']}"
            # )


def get_teacher_stats(class_id: str) -> TeacherStats:
    recordings = (
        st.session_state["conn"]
        .table("recordings")
        .select("*")
        .eq("user_id", st.session_state["user"].id)
        .eq("class_id", class_id)
        .execute()
    )
    if not recordings.data:
        return
    df = pd.DataFrame(recordings.data)
    df["has_transcript"] = [r["transcript"] is not None for r in recordings.data]
    name = (
        st.session_state.user.user_metadata["first_name"]
        + " "
        + st.session_state.user.user_metadata["last_name"]
    ).lower()

    rps = [
        RecordingProcessor(
            id=rec["id"],
            ts=parser.isoparse(rec["date"]),
            transcript=rec["transcript"],
            chatbot=OpenAIChatbot(model_id="gpt-4-turbo", temperature=0.0),
            db_client=st.session_state["conn"],
            teacher=name,
        )
        for _, rec in df.iterrows()
        if rec["has_transcript"]
    ]

    if not rps:
        return
    for rp in rps:
        rp.process()

    return TeacherStats(name=name, recording_processors=rps)


def sentiment_analysis(teacher_stats: TeacherStats):
    st.divider()


def dashboard():
    cl = st.sidebar.radio(
        "Classes Taught", st.session_state.class_name_id_mapping.keys()
    )
    class_id = st.session_state.class_name_id_mapping[cl]
    if "teacher_stats" not in st.session_state:
        st.session_state["teacher_stats"] = {class_id: get_teacher_stats(class_id)}
    elif class_id not in st.session_state.teacher_stats:
        st.session_state.teacher_stats[class_id] = get_teacher_stats(class_id)

    with st.sidebar:
        dashboard_option = option_menu(
            "Metrics",
            [
                "Participation",
                "Pairwise Participation",
                "Teacher Interruption",
                "Sentiment Analysis",
            ],
            icons=[
                "person-raised-hand",
                "people-fill",
                "person-arms-up",
                "emoji-heart-eyes",
            ],
        )
    match dashboard_option:
        case "Participation":
            render_participation_charts(st.session_state.teacher_stats[class_id])
        case "Pairwise Participation":
            render_pairwise_charts(st.session_state.teacher_stats[class_id])
        case "Teacher Interruption":
            render_interruption_charts(st.session_state.teacher_stats[class_id])
        case "Sentiment Analysis":
            sentiment_analysis(st.session_state.teacher_stats[class_id])


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
        initialize_state()
        pg = st.navigation(
            [
                st.Page(
                    dashboard,
                    title="Dashboard",
                    icon=":material/dashboard:",
                    default=True,
                ),
                st.Page(
                    add_recording, title="Add recording", icon=":material/videocam:"
                ),
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
