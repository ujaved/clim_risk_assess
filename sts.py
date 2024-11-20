import streamlit as st
from recording_processor import RecordingProcessor, MetricName
from teacher_stats import TeacherStats
from chatbot import OpenAIChatbot
from dateutil import parser
import altair as alt
from gotrue.errors import AuthApiError
import pandas as pd
from altair import datum
from streamlit_option_menu import option_menu
import statsmodels.api as sm
from scipy.stats import norm
from streamlit_url_fragment import get_fragment
from store import DBClient
import jwt
from io import BytesIO
from utils import get_s3_object_keys
import boto3
import os
from chatbot import (
    EmotionAnalysis,
    ModeAnalysis,
    SecondaryToPrimaryMapping,
    PrimaryToSecondaryMapping,
)
from collections import defaultdict
from itertools import groupby
from utils import num_secs
import random


WELCOME_MSG = "Welcome to Scientifically Taught Science"


# Initialize connection.
# @st.cache_resource
def init_connection() -> None:
    if "db_client" not in st.session_state:
        st.session_state["db_client"] = DBClient(
            st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
        )
    if "s3_client" not in st.session_state:
        st.session_state["s3_client"] = boto3.client("s3")


def initialize_state():
    if "class_name_id_mapping" in st.session_state:
        return
    classes = st.session_state.db_client.get_classes(st.session_state.user.id)
    st.session_state["class_name_id_mapping"] = {cl["name"]: cl["id"] for cl in classes}


def add_recording_cb():

    if (
        not st.session_state.get("recording_url")
        or not st.session_state.get("recording_date")
        or not st.session_state.get("recording_class")
    ):
        st.error("Input data is missing")
        return
    try:
        st.session_state.db_client.insert_recording(
            st.session_state.user.id,
            st.session_state.recording_url,
            st.session_state.recording_date,
            st.session_state["class_name_id_mapping"][
                st.session_state.get("recording_class")
            ],
        )
        st.success("Recording added successfully")
    except Exception as e:
        st.error(str(e))


def new_class_cb():
    st.session_state.db_client.insert_class(
        st.session_state.new_class, st.session_state.user.id, st.session_state.org_id
    )
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


def get_teacher_stats(class_id: str) -> TeacherStats | None:
    recordings = st.session_state.db_client.get_recordings(
        st.session_state["user"].id, class_id
    )
    if not recordings:
        return None
    df = pd.DataFrame(recordings)
    df["has_transcript"] = [r["transcript"] is not None for r in recordings]
    name = (
        st.session_state.user.user_metadata["first_name"]
        + " "
        + st.session_state.user.user_metadata["last_name"]
    ).lower()

    speakers = st.session_state.db_client.get_speakers(class_id)
    name_mapping = {
        a: s["name"] for s in speakers if s["alt_names"] for a in s["alt_names"]
    }

    rps = [
        RecordingProcessor(
            id=rec["id"],
            ts=parser.isoparse(rec["date"]),
            transcript=rec["transcript"],
            chatbot=OpenAIChatbot(model_id="gpt-4o-2024-08-06", temperature=0.0),
            db_client=st.session_state.db_client,
            teacher=name,
            name_mapping=name_mapping,
        )
        for _, rec in df.iterrows()
        if rec["has_transcript"]
    ]

    if not rps:
        return None
    for rp in rps:
        rp.process()

    return TeacherStats(name=name, recording_processors=rps, name_mapping=name_mapping)


def emotion_analysis(teacher_stats: TeacherStats):
    dates = sorted([rp.ts.date() for rp in teacher_stats.recording_processors])
    col1, col2 = st.columns(2)
    start_date = col1.date_input(
        "Start date",
        value=dates[0],
        min_value=dates[0],
        max_value=dates[-1],
    )
    end_date = col2.date_input(
        "End date",
        value=dates[0],
        min_value=dates[0],
        max_value=dates[-1],
    )

    if start_date > end_date:
        st.error("start date cannot be later than the end date")
        return

    rps = [
        rp
        for rp in teacher_stats.recording_processors
        if rp.ts.date() >= start_date and rp.ts.date() <= end_date
    ]
    duration_mins = int(max([rp.class_duration_secs for rp in rps]) / 60)

    cur_val = st.session_state.get("emotion_analysis_interval")
    interval = st.slider(
        "Emotion analysis interval (minutes)",
        value=cur_val or 2,
        min_value=2,
        max_value=int(duration_mins / 2),
        step=int(duration_mins / 10),
        key="emotion_analysis_interval",
    )

    emotion_counts = defaultdict(lambda: defaultdict(int))
    primary_emotion_counts = defaultdict(lambda: defaultdict(int))
    emotion_analysis_json_by_date = {}
    num_intervals = 0
    for rp in rps:
        emotion_analysis_json = st.session_state.db_client.get_emotion_analysis(
            rp.id, interval
        )
        if emotion_analysis_json:
            emotion_analysis = EmotionAnalysis(**emotion_analysis_json)
        else:
            emotion_analysis = rp.get_emotion_analysis(interval)
            emotion_analysis_json = emotion_analysis.model_dump()
            st.session_state.db_client.insert_emotion_analysis(
                rp.id, interval, emotion_analysis_json
            )
        num_intervals += len(emotion_analysis.emotions)
        for s in emotion_analysis.emotions:
            primary_emotions = {SecondaryToPrimaryMapping[l.value] for l in s.labels}
            for p in primary_emotions:
                primary_emotion_counts[rp.ts.date().isoformat()][p] += 1
            for l in s.labels:
                emotion_counts[rp.ts.date().isoformat()][l.value] += 1

        emotion_analysis_json_by_date[rp.ts.date().isoformat()] = emotion_analysis_json

    emotion_data = [
        {
            "date": date,
            "emotion": s,
            "count": c,
            "num_intervals": num_intervals,
        }
        for date, sc in emotion_counts.items()
        for s, c in sc.items()
    ]
    emotion_data_df = pd.DataFrame(emotion_data)

    primary_emotion_data = [
        {
            "date": date,
            "emotion": s,
            "count": c,
            "num_intervals": num_intervals,
        }
        for date, sc in primary_emotion_counts.items()
        for s, c in sc.items()
    ]
    primary_emotion_data_df = pd.DataFrame(primary_emotion_data)

    chart = (
        alt.Chart(
            primary_emotion_data_df.groupby(["emotion", "num_intervals"])
            .sum("count")
            .reset_index()
        )
        .transform_calculate(percent="datum.count*100/datum.num_intervals")
        .mark_bar()
        .encode(
            x=alt.X(
                "emotion:O",
                sort=alt.EncodingSortField(field="percent", order="descending"),
            ),
            y="percent:Q",
            color=alt.Color("emotion", scale=alt.Scale(scheme="dark2")),
        )
        .add_params(alt.selection_point())
    )
    st.subheader("Aggregate Primary Emotions", divider=True)
    st.info("Click on a bar to see the aggregate for the secondary emotions")
    selection = st.altair_chart(
        chart, use_container_width=True, on_select="rerun"
    ).selection.param_1
    if selection:
        primary = selection[0]["emotion"]
        secondary = PrimaryToSecondaryMapping[primary]
        emotion_data_df = emotion_data_df[emotion_data_df["emotion"].isin(secondary)]
        chart = (
            alt.Chart(
                emotion_data_df.groupby(["emotion", "num_intervals"])
                .sum("count")
                .reset_index()
            )
            .transform_calculate(percent="datum.count*100/datum.num_intervals")
            .mark_bar()
            .encode(
                x=alt.X(
                    "emotion:O",
                    sort=alt.EncodingSortField(field="percent", order="descending"),
                ),
                y="percent:Q",
                color=alt.Color("emotion", scale=alt.Scale(scheme="dark2")),
            )
        )
        st.subheader(f"Aggregate Secondary Emotions for {primary}", divider=True)
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Emotion Analysis Timelines", divider=True)
    for d, j in emotion_analysis_json_by_date.items():
        st.info(d)
        st.table(j["emotions"])


def mode_analysis(teacher_stats: TeacherStats):
    dates = sorted([rp.ts.date() for rp in teacher_stats.recording_processors])
    col1, col2 = st.columns(2)
    start_date = col1.date_input(
        "Start date",
        value=dates[0],
        min_value=dates[0],
        max_value=dates[-1],
    )
    end_date = col2.date_input(
        "End date",
        value=dates[0],
        min_value=dates[0],
        max_value=dates[-1],
    )

    if start_date > end_date:
        st.error("start date cannot be later than the end date")
        return

    rps = [
        rp
        for rp in teacher_stats.recording_processors
        if rp.ts.date() >= start_date and rp.ts.date() <= end_date
    ]
    duration_mins = int(max([rp.class_duration_secs for rp in rps]) / 60)

    cur_val = st.session_state.get("mode_analysis_interval")
    interval = st.slider(
        "Mode analysis interval (minutes)",
        value=cur_val or 2,
        min_value=2,
        max_value=int(duration_mins / 3),
        step=int(duration_mins / 10),
        key="mode_analysis_interval",
    )

    num_intervals = 0
    mode_counts = defaultdict(lambda: defaultdict(int))
    mode_analysis_by_date = {}
    modes = []
    for rp in rps:
        mode_analysis_json = st.session_state.db_client.get_mode_analysis(
            rp.id, interval
        )
        if mode_analysis_json:
            mode_analysis = ModeAnalysis(**mode_analysis_json)
        else:
            mode_analysis = rp.get_mode_analysis(interval)
            mode_analysis_json = mode_analysis.model_dump()
            st.session_state.db_client.insert_mode_analysis(
                rp.id, interval, mode_analysis_json
            )
        num_intervals += len(mode_analysis.modes)
        for m in mode_analysis.modes:
            modes.append((rp.date, m))
            mode_counts[rp.date][m.label.value] += 1

        mode_analysis_by_date[rp.date] = mode_analysis_json

        # modes.sort(key=lambda m: m[1].label)
        # to_label = [random.choice(list(v)) for _, v in groupby(modes, lambda m: m[1].label)]

    mode_data = [
        {
            "date": date,
            "mode": m,
            "count": c,
            "num_intervals": len(mode_analysis_by_date[date]["modes"]),
            "tot_num_intervals": num_intervals,
        }
        for date, mc in mode_counts.items()
        for m, c in mc.items()
    ]
    mode_data_df = pd.DataFrame(mode_data)
    chart = (
        alt.Chart(
            mode_data_df.groupby(["mode", "tot_num_intervals"])
            .sum("count")
            .reset_index()
        )
        .transform_calculate(percent="datum.count*100/datum.tot_num_intervals")
        .mark_bar()
        .encode(
            x=alt.X(
                "mode:O",
                sort=alt.EncodingSortField(field="percent", order="descending"),
            ),
            y="percent:Q",
            color=alt.Color("mode", scale=alt.Scale(scheme="dark2")),
        )
    )
    st.subheader("Aggregate Modes", divider=True)
    st.altair_chart(chart, use_container_width=True)

    chart = (
        alt.Chart(mode_data_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q").stack("normalize").title(None),
            y="date:O",
            color=alt.Color("mode", scale=alt.Scale(scheme="dark2")),
            order=alt.Order("count:Q", sort="descending"),
        )
        .add_params(alt.selection_point())
    )
    st.subheader("Modes for each recording", divider=True)
    st.info("Click on a segment to see details")
    selection = st.altair_chart(
        chart, use_container_width=True, on_select="rerun"
    ).selection.param_1
    if selection:
        date = selection[0]["date"]
        mode = selection[0]["mode"]
        labeled_modes = [
            m
            for m in mode_analysis_by_date[date]["modes"]
            if m["label"] == selection[0]["mode"]
        ]
        st.info(
            f"Intervals on {date} labeled as {mode}. Select a row to view the entire transcript snippet"
        )
        row_selected = st.dataframe(
            labeled_modes,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        ).selection.rows
        if row_selected:
            row = row_selected[0]
            start_secs = num_secs(labeled_modes[row]["start_time"])
            end_secs = num_secs(labeled_modes[row]["end_time"])
            rp = [rp for rp in rps if rp.date == date][0]
            snippet = [
                (d[0], d[1])
                for d in rp.dialogue
                if d[2] >= start_secs and d[2] < end_secs
            ]
            for d in snippet:
                if d[0] == rp.teacher:
                    with st.chat_message("ai"):
                        st.write(d[0] + ": " + d[1])
                else:
                    with st.chat_message("user"):
                        st.write(d[0] + ": " + d[1])


def facial_recognition(class_id: str, teacher_stats: TeacherStats):
    recording_ids = {rp.id: rp.ts.date() for rp in teacher_stats.recording_processors}
    recording_stats = st.session_state.db_client.get_recording_stats(
        recording_ids.keys()
    )
    fc_recording_stats = {
        s["recording_id"]: s["facial_recognition_intervals"]
        for s in recording_stats
        if s.get("facial_recognition_intervals")
    }
    if not fc_recording_stats:
        st.info("No facial recognition metrics")
    speakers = st.session_state.db_client.get_speakers(class_id)
    speakers = {s["id"]: s["name"] for s in speakers}
    for recording_id, fc in fc_recording_stats.items():
        st.subheader(recording_ids[recording_id])

        df_data = []
        for speaker_id, intervals in fc.items():
            for i in intervals:
                for second in range(i[0], i[1] + 1, 30):
                    df_data.append(
                        {
                            "speaker": speakers[speaker_id],
                            "minutes_into_video": float(second) / 60,
                        }
                    )
        df = pd.DataFrame(df_data)

        chart = (
            alt.Chart(df)
            .mark_point(size=2)
            .encode(x="minutes_into_video:Q", y="speaker:N", color="speaker")
        )
        st.altair_chart(chart, use_container_width=True)


def label_student_face_cb(class_id: str, key: str, image_s3_key: str):
    student = st.session_state[key]
    st.session_state.db_client.insert_speaker(class_id, student, image_s3_key)
    st.session_state[key] = None


def name_mapping_cb(
    class_id: str, new_name_mapping: pd.DataFrame, current_mapping: dict[str, str]
):
    for idx, edited_fields in st.session_state.new_name_mapping_data_editor[
        "edited_rows"
    ].items():
        nick = new_name_mapping[idx]["transcript name"]
        if edited_fields.get("new student ?"):
            st.session_state.db_client.insert_speaker(
                class_id, name=nick, alt_names=[nick]
            )
        elif edited_fields.get("mapping options"):
            name = edited_fields.get("mapping options")
            curr_alt_names = [k for k, v in current_mapping.items() if v == name]
            st.session_state.db_client.update_speaker(
                class_id, name=name, alt_names=curr_alt_names + [nick]
            )

    # remove teacher stats so that it is refreshed with the new name mappings
    del st.session_state.teacher_stats[class_id]


def label_speakers():
    class_id = get_class_id()
    if "teacher_stats" not in st.session_state:
        st.session_state["teacher_stats"] = {}
    st.session_state.teacher_stats[class_id] = get_teacher_stats(class_id)

    teacher_stats = st.session_state.teacher_stats[class_id]

    cur_name_mapping = [
        {"transcript name": nick, "actual name": name, "remove": False}
        for nick, name in teacher_stats.name_mapping.items()
    ]
    new_names = [
        name
        for name in teacher_stats.students
        if name not in teacher_stats.name_mapping
    ]
    if cur_name_mapping:
        st.subheader("Current name mappings", divider=True)
        st.data_editor(
            cur_name_mapping,
            hide_index=True,
            use_container_width=True,
            disabled=["actual name", "transcript name", "remove"],
        )
    if new_names:
        st.subheader("New name mappings", divider=True)
        new_name_mapping = [
            {"transcript name": name, "new student ?": False, "mapping options": ""}
            for name in new_names
        ]
        df = pd.DataFrame(new_name_mapping)
        df["mapping options"] = df["mapping options"].astype("category")
        df["mapping options"] = df["mapping options"].cat.add_categories(
            set(teacher_stats.name_mapping.values())
        )
        st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            on_change=name_mapping_cb,
            args=(class_id, new_name_mapping, teacher_stats.name_mapping),
            key="new_name_mapping_data_editor",
            disabled=["transcript name"],
        )

    speakers = st.session_state.db_client.get_speakers(class_id)
    labeled_speakers = {s["name"]: s["s3_key"] for s in speakers if s["s3_key"]}
    s3_keys = {s["s3_key"] for s in speakers}
    num_columns = 2

    st.subheader("Labeled Speaker faces", divider=True)
    cols = st.columns(num_columns)
    for idx, s in enumerate(labeled_speakers.items()):
        col = cols[idx % num_columns]
        image = BytesIO(
            st.session_state.s3_client.get_object(
                Bucket=os.getenv("S3_BUCKET"), Key=s[1]
            )["Body"].read()
        )
        with col:
            st.image(image)
            st.write(s[0])

    st.subheader("Missing labels", divider=True)
    image_files = get_s3_object_keys(st.session_state.s3_client, class_id)
    missing_labels = [i for i in image_files if i not in s3_keys]
    if not missing_labels:
        st.info("No missing labels for this class")
        return

    if "teacher_stats" not in st.session_state:
        st.session_state["teacher_stats"] = {class_id: get_teacher_stats(class_id)}
    elif class_id not in st.session_state.teacher_stats:
        st.session_state.teacher_stats[class_id] = get_teacher_stats(class_id)

    teacher_stats = st.session_state.teacher_stats[class_id]
    cols = st.columns(num_columns)
    for idx, m in enumerate(missing_labels):
        col = cols[idx % num_columns]
        image = BytesIO(
            st.session_state.s3_client.get_object(Bucket=os.getenv("S3_BUCKET"), Key=m)[
                "Body"
            ].read()
        )
        with col:
            st.image(image)
            key = f"image_label_{class_id}_{idx}"
            st.selectbox(
                "Student",
                [teacher_stats.name] + list(teacher_stats.students),
                index=None,
                key=key,
                on_change=label_student_face_cb,
                args=(class_id, key, m),
            )


def get_class_id() -> str | None:
    idx = 0
    if "class_selected" in st.session_state:
        names = list(st.session_state.class_name_id_mapping.keys())
        for i, cl in enumerate(names):
            if st.session_state.class_selected == cl:
                idx = i
                break

    st.sidebar.radio(
        "Classes Taught",
        list(st.session_state.class_name_id_mapping.keys()),
        index=idx,
        key="class_selected",
    )
    return st.session_state.class_name_id_mapping.get(st.session_state.class_selected)


def dashboard():
    class_id = get_class_id()
    if class_id is None:
        return
    if "teacher_stats" not in st.session_state:
        st.session_state["teacher_stats"] = {class_id: get_teacher_stats(class_id)}
    elif st.session_state.teacher_stats.get(class_id) is None:
        st.session_state.teacher_stats[class_id] = get_teacher_stats(class_id)

    with st.sidebar:
        dashboard_option = option_menu(
            "Metrics",
            [
                "Participation",
                "Pairwise Participation",
                "Comparison",
                "Facial Recognition",
                "Emotion Analysis",
                "Mode Analysis",
            ],
            icons=[
                "person-raised-hand",
                "people-fill",
                "bar-chart-steps",
                "person-bounding-box",
                "emoji-heart-eyes",
                "upc-scan",
            ],
        )

    if st.session_state.teacher_stats.get(class_id) is None:
        return
    match dashboard_option:
        case "Participation":
            render_participation_charts(st.session_state.teacher_stats[class_id])
        case "Pairwise Participation":
            render_pairwise_charts(st.session_state.teacher_stats[class_id])
        case "Comparison":
            metric_comparison(st.session_state.teacher_stats[class_id])
        case "Facial Recognition":
            facial_recognition(class_id, st.session_state.teacher_stats[class_id])
        case "Emotion Analysis":
            emotion_analysis(st.session_state.teacher_stats[class_id])
        case "Mode Analysis":
            mode_analysis(st.session_state.teacher_stats[class_id])


def set_org_id():
    orgs = st.session_state.db_client.get_orgs()
    for org in orgs:
        if (
            org["name"].lower()
            == st.session_state.user.user_metadata["organization"].lower()
        ):
            st.session_state["org_id"] = org["id"]
            break

    if "org_id" not in st.session_state:
        st.session_state.db_client.insert_org(
            st.session_state.user.user_metadata["organization"]
        )
        set_org_id()


def login_submit(is_login: bool):
    if is_login:
        if not st.session_state.login_email or not st.session_state.login_password:
            st.error("Please provide login information")
            return
        try:
            st.session_state["user"] = st.session_state.db_client.sign_in(
                st.session_state.login_email, st.session_state.login_password
            )
            st.session_state["authenticated"] = True
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
        st.session_state.db_client.invite_user_by_email(
            st.session_state.register_email,
            st.session_state.register_first_name,
            st.session_state.register_last_name,
            st.session_state.register_org,
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
        st.session_state["user"] = st.session_state.db_client.update_user_password(
            user_id, st.session_state.reset_password_password
        )
        st.session_state["authenticated"] = True
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
    init_connection()

    add_recording_pg = st.Page(
        add_recording, title="Add recording", icon=":material/videocam:"
    )
    dashboard_pg = st.Page(
        dashboard, title="Dashboard", icon=":material/dashboard:", default=True
    )
    label_speakers_pg = st.Page(
        label_speakers,
        title="Label Student Names and Faces",
        icon=":material/familiar_face_and_zone:",
    )

    if st.session_state.get("authenticated"):
        initialize_state()
        pg = st.navigation([dashboard_pg, add_recording_pg, label_speakers_pg])
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
