import streamlit as st
from recording_processor import RecordingProcessor, MetricName
from teacher_stats import TeacherStats
from chatbot import OpenAIChatbot
from dateutil import parser
import altair as alt
from supabase import create_client, Client
from gotrue.errors import AuthApiError
import pandas as pd
from altair import datum
from streamlit_option_menu import option_menu
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from streamlit_url_fragment import get_fragment
import jwt


WELCOME_MSG = "Welcome to Scientifically Taught Science"


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
            .eq("teacher_id", st.session_state.user.id)
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
            "org_id": st.session_state.org_id,
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
    # st.subheader("Pairwise participation for a (lead, follow) pair, given the data, is the conditional probability that the follow speaks right after the lead, divided by random chance that the follow speaks. ",divider=True)
    st.subheader(
        "Pairwise participation for a (lead, follow) pair is a statistically significant Pearson adjusted standardized residual (chi-squared test)",
        divider=True,
    )
    with st.container(border=True):
        col1, col2 = st.columns(2)
        df = teacher_stats.get_pairwise_following_df(
            col1.date_input("start date", value=None),
            col2.date_input("end date", value=None),
        )
        contingency_table = []
        students = sorted(teacher_stats.students)
        for lead in students:
            row = []
            for follow in students:
                if lead == follow:
                    continue
                count = 0
                if ((df["lead"] == lead) & (df["follow"] == follow)).any():
                    count = df[(df["lead"] == lead) & (df["follow"] == follow)].iloc[0][
                        "count"
                    ]
                row.append(count)
            contingency_table.append(row)
        table = sm.stats.Table(contingency_table)
        for i, lead_resids in enumerate(table.standardized_resids):
            lead = students[i]
            for j, resid in enumerate(lead_resids):
                follow = students[j] if j < i else students[j + 1]
                df.loc[
                    (df["lead"] == lead) & (df["follow"] == follow),
                    "standardized_resid",
                ] = resid
        num_cells = len(students) * (len(students) - 1)

        a = "\alpha"
        alpha = st.slider(
            "Chi-square significance $a$",
            value=0.05,
            min_value=0.01,
            max_value=0.1,
            step=0.01,
        )
        alpha_bon = alpha / num_cells
        critical_value = norm.ppf(1 - alpha_bon / 2)
        df = df.loc[abs(df["standardized_resid"]) >= critical_value]
        # df.loc[abs(df["standardized_resid"]) < critical_value, "standardized_resid"] = 0.0

        st.info(
            "Click on a cell to get a concrete description for the (lead,follow) participation metric"
        )

        matrix = (
            alt.Chart(df)
            .mark_rect(color="orange", size=10)
            .encode(
                alt.X("follow"),
                alt.Y("lead"),
                alt.Color("standardized_resid")
                .scale(scheme="turbo")
                .title(
                    [
                        "Pearson standardized",
                        "residual value",
                        "(statistically significant)",
                    ]
                ),
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
            a = "\alpha"
            a_b = "\alpha_{b}"
            N_0_1 = "\mathcal{N}(0,1)_{1-a_b/2}"

            prob_explanation = f"""
            Excluding teacher utterances, the probability of follow **{follow}** speaking $P_f$ is {row['prob_turns_follow']:.4f}.
            The probability of lead **{lead}** speaking $P_l$ is {row['prob_turns_lead']:.4f}.
            The empirical probability of the pair **({lead}-{follow})** ${p_lf}$ is (number of pair occurences)/(number of total pair occurences) = {row['prob_pairwise']:.4f}.
            The conditional probability of **{follow}**'s turn after **{lead}** ${p_f_l}$ = ${p_lf}$/$P_l$ = {row['prob_pairwise']:.4f}/{row['prob_turns_lead']:.4f} = {row['prob_cond']:.4f}.
            The pairwise participation ratio is ${p_f_l}$ divided by the random chance of follow's turn, i.e., ${p_f_l}$/$P_f$ = {row['prob_cond']:.4f}/{row['prob_turns_follow']:.4f} = {row['ratio']:.4f}.
            """
            pearson_explanation = f"""
            Excluding teacher utterances, the number of **({lead}-{follow})** occurences is {row['count']}.
            For a {len(students)}x{len(students)-1} contingency table of lead-follow pair utterances, the Pearson adjusted standardized residual is {row['standardized_resid']:.2f}.
            With the statistical significance $a$ = {alpha}, with the Bonferroni correction $a_b$ = {alpha}/{num_cells} = {alpha_bon:.4f}, the 
            corresponsing critical value is ${N_0_1}$ = {critical_value:.2f}.
            Hence {row['standardized_resid']:.2f} is  significantly different than expected under the null hypothesis of no assocation.       
            """
            st.write(pearson_explanation)


def metric_comparison_cb(teacher_stats: TeacherStats, container):
    if st.session_state.start_date_baseline > st.session_state.end_date_baseline:
        st.error("Baseline start date cannot be later than the end date")
        return
    if st.session_state.start_date_comparison > st.session_state.end_date_comparison:
        st.error("Comparison start date cannot be later than the end date")
        return
    data = []
    for s in st.session_state.comprison_speakers:
        mean_baseline = 0.0
        mean_comparison = 0.0
        num_baseline = 0
        num_comparison = 0
        for rp in teacher_stats.recording_processors:
            if s not in rp.speaker_stats:
                continue
            speaker_stats = rp.speaker_stats[s].transform(rp.class_duration_secs)
            if (
                rp.ts.date() >= st.session_state.start_date_baseline
                and rp.ts.date() <= st.session_state.end_date_baseline
            ):
                mean_baseline += speaker_stats[st.session_state.comparison_metric]
                num_baseline += 1

            if (
                rp.ts.date() >= st.session_state.start_date_comparison
                and rp.ts.date() <= st.session_state.end_date_comparison
            ):
                mean_comparison += speaker_stats[st.session_state.comparison_metric]
                num_comparison += 1
        mean_baseline /= num_baseline
        mean_comparison /= num_comparison
        perc_change = float(mean_comparison - mean_baseline) * 100 / mean_baseline
        data.append(
            {
                "mean baseline value": mean_baseline,
                "mean value to compare": mean_comparison,
                "% change": perc_change,
            }
        )
    if data:
        with container:
            st.dataframe(
                pd.DataFrame(data, index=st.session_state.comprison_speakers),
                use_container_width=True,
            )


def metric_comparison(teacher_stats: TeacherStats):
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    data_container = st.container()
    col1.multiselect(
        "Select a speaker or list of speakers to get comparison statistics",
        [teacher_stats.name] + list(teacher_stats.students),
        key="comprison_speakers",
        placeholder="Choose speaker/s",
        on_change=metric_comparison_cb,
        args=(teacher_stats, data_container),
    )
    col2.radio(
        "Metric",
        [m.value for m in MetricName],
        key="comparison_metric",
        on_change=metric_comparison_cb,
        args=(teacher_stats, data_container),
    )

    dates = sorted([rp.ts.date() for rp in teacher_stats.recording_processors])
    col3.date_input(
        "Start date for baseline range",
        key="start_date_baseline",
        value=dates[0],
        min_value=dates[0],
        max_value=dates[-1],
        on_change=metric_comparison_cb,
        args=(teacher_stats, data_container),
    )
    col3.date_input(
        "End date for baseline range",
        key="end_date_baseline",
        value=dates[0],
        min_value=dates[0],
        max_value=dates[-1],
        on_change=metric_comparison_cb,
        args=(teacher_stats, data_container),
    )
    col4.date_input(
        "Start date for comparison range",
        key="start_date_comparison",
        value=dates[-1],
        min_value=dates[0],
        max_value=dates[-1],
        on_change=metric_comparison_cb,
        args=(teacher_stats, data_container),
    )
    col4.date_input(
        "End date for comparison range",
        key="end_date_comparison",
        value=dates[-1],
        min_value=dates[0],
        max_value=dates[-1],
        on_change=metric_comparison_cb,
        args=(teacher_stats, data_container),
    )


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
    if cl is None:
        st.warning("No class or recording yet added")
        return
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
                "Comparison",
                "Sentiment Analysis",
            ],
            icons=[
                "person-raised-hand",
                "people-fill",
                "bar-chart-steps",
                "emoji-heart-eyes",
            ],
        )
    match dashboard_option:
        case "Participation":
            render_participation_charts(st.session_state.teacher_stats[class_id])
        case "Pairwise Participation":
            render_pairwise_charts(st.session_state.teacher_stats[class_id])
        case "Comparison":
            metric_comparison(st.session_state.teacher_stats[class_id])
        case "Sentiment Analysis":
            sentiment_analysis(st.session_state.teacher_stats[class_id])


def set_org_id():
    orgs = st.session_state.conn.table("organizations").select("*").execute()
    for org in orgs.data:
        if (
            org["name"].lower()
            == st.session_state.user.user_metadata["organization"].lower()
        ):
            st.session_state["org_id"] = org["id"]
            break

    if "org_id" not in st.session_state:
        st.session_state.conn.table("organizations").insert(
            {"name": st.session_state.user.user_metadata["organization"]}
        ).execute()
        set_org_id()


def login_submit(is_login: bool):
    if is_login:
        if not st.session_state.login_email or not st.session_state.login_password:
            st.error("Please provide login information")
            return
        try:
            data = st.session_state.conn.auth.sign_in_with_password(
                {
                    "email": st.session_state.login_email,
                    "password": st.session_state.login_password,
                }
            )
            st.session_state["authenticated"] = True
            st.session_state["user"] = data.user
            set_org_id()
        except AuthApiError as e:
            st.error(e)
        return

    try:
        if (
            not st.session_state.register_email
            or not st.session_state.register_first_name
            or not st.session_state.register_last_name
            or not st.session_state.register_org
        ):
            st.error("Please provide all requested information")
            return
        st.session_state.conn.auth.admin.invite_user_by_email(
            st.session_state.register_email,
            options={
                "data": {
                    "first_name": st.session_state.register_first_name,
                    "last_name": st.session_state.register_last_name,
                    "organization": st.session_state.register_org,
                }
            },
        )
        st.info("An email invite has been sent to your email")
    except AuthApiError as e:
        st.error(e)


def reset_password_submit(user_id: str):
    if (
        not st.session_state.reset_password_password
        or not st.session_state.reset_password_confirm_password
    ):
        st.error("Please enter password and confirm password")
        return
    if (
        st.session_state.reset_password_password
        != st.session_state.reset_password_confirm_password
    ):
        st.error("Passwords don't match")
        return
    try:
        data = st.session_state.conn.auth.admin.update_user_by_id(
            user_id, {"password": st.session_state.reset_password_password}
        )
        st.session_state["authenticated"] = True
        st.session_state["user"] = data.user
        set_org_id()
    except AuthApiError as e:
        st.error(e)


def reset_password(email: str, user_id: str):
    with st.form("login_form", clear_on_submit=True):
        st.text_input("Email", key="reset_password_email", value=email, disabled=True)
        st.text_input("Password", type="password", key="reset_password_password")
        st.text_input(
            "Confirm Password", type="password", key="reset_password_confirm_password"
        )
        st.form_submit_button("Submit", on_click=reset_password_submit, args=(user_id,))


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
            st.text_input("First name", key="register_first_name")
            st.text_input("Last name", key="register_last_name")
            st.text_input("Organization", key="register_org")
            st.form_submit_button("Submit", on_click=login_submit, args=[False])


def main():

    st.set_page_config(page_title="STS", page_icon=":teacher:", layout="wide")
    st.session_state["conn"] = init_connection()

    add_recording_pg = st.Page(
        add_recording, title="Add recording", icon=":material/videocam:"
    )
    dashboard_pg = st.Page(
        dashboard, title="Dashboard", icon=":material/dashboard:", default=True
    )

    if st.session_state.get("authenticated"):
        initialize_state()
        pg = st.navigation([dashboard_pg, add_recording_pg])
        pg.run()
    elif "reset_password" in st.query_params:
        fragment = get_fragment()
        acces_token = (fragment.split("access_token=")[1]).split("&")[0]
        payload = jwt.decode(acces_token, options={"verify_signature": False})
        reset_password(payload["email"], payload["sub"])
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
