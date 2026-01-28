"""Code to analyze app rating responses."""

import numpy as np

from modules.definitions.types import StudyGroup, Survey
from modules.utils.comparisons import plot_per_question
from modules.utils.output_formatting import format_float
from modules.utils.radar_chart import (
    create_radar_chart,
    get_radar_chart_data,
    get_radar_chart_mean,
    get_radar_chart_means,
)

UMARS_SUBSCALES = {
    "Engagement": [
        "Entertainment",
        "Interest",
        "Customisation",
        "Interactivity",
        "Target group",
    ],
    "Functionality": [
        "Performance",
        "Ease of use",
        "Navigation",
        "Gestural design",
    ],
    "Aesthetics": [
        "Layout",
        "Graphics",
        "Visual appeal",
    ],
    "Information": [
        "Quality of information",
        "Quantity of information",
        "Visual information",
        "Credibility of source",
    ],
    "Subjective items": [
        "Would you recommend this app to people who might benefit from it?",
        (
            "How many times do you think you would use this app in the next 12 "
            "months if it was relevant to you?"
        ),
        "Would you pay for this app?",
        "What is your overall (star) rating of the app?",
    ],
    "Perceived impact": [
        "Awareness",
        "Knowledge",
        "Attitudes",
        "Intention to change",
        "Help seeking",
        "Behaviour change",
    ],
}
OVERALL_SCORE_SUBSCALES = [
    "Engagement",
    "Functionality",
    "Aesthetics",
    "Information",
]
NO_SPIDER_CHART_SUBSCALES = ["Subjective items"]
NO_SPIDER_CHART_COLUMNS = {
    "Perceived impact": ["Further comments about the app?"],
}


def get_overall_app_rating_data() -> tuple[
    list[list[float]],
    list[StudyGroup],
    list[str],
]:
    """Get rating data per participant per subscale."""
    survey = Survey.APP_RATING
    dimensions = []
    participant_data = []
    study_groups = None
    participant_ids = None
    for subscale_name in OVERALL_SCORE_SUBSCALES:
        subscale_columns = UMARS_SUBSCALES[subscale_name]
        subscale_data, study_groups, participant_ids = get_radar_chart_data(
            survey,
            subscale_columns,
        )
        subscale_means = [
            get_radar_chart_mean(participant_subscale_data, zeros_are_na=True)
            for participant_subscale_data in subscale_data
        ]
        subscale_mean = get_radar_chart_mean(
            subscale_means,
            zeros_are_na=False,
        )
        dimensions.append(
            f"{subscale_name}\n({format_float(subscale_mean)})",
        )
        participant_data.append(subscale_means)
    return np.array(participant_data).T, study_groups, participant_ids


def analyze_app_rating() -> None:
    """Analyze app rating."""
    survey = Survey.APP_RATING
    overall_rating_data, overall_study_groups, _ = get_overall_app_rating_data()
    subscale_mean_list = get_radar_chart_means(
        overall_rating_data,
        OVERALL_SCORE_SUBSCALES,
        zeros_are_na=False,
    )
    overall_mean = get_radar_chart_mean(
        subscale_mean_list,
        zeros_are_na=False,
    )
    title_addition = (
        f"\n(n = {len(overall_rating_data)}, "
        f"mean: {format_float(overall_mean)})"
    )
    create_radar_chart(
        survey=survey,
        data=overall_rating_data,
        subscale="overall",
        columns=OVERALL_SCORE_SUBSCALES,
        study_groups=overall_study_groups,
        zeros_are_na=False,
        title_addition=title_addition,
    )
    subscale_means = {}
    for index, subscale_mean in enumerate(subscale_mean_list):
        subscale_means[OVERALL_SCORE_SUBSCALES[index]] = format_float(
            subscale_mean,
        )
    for subscale_name, subscale_columns in UMARS_SUBSCALES.items():
        if subscale_name not in NO_SPIDER_CHART_SUBSCALES:
            data, study_groups, _ = get_radar_chart_data(
                survey,
                subscale_columns,
            )
            title_addition = f"\n(n = {len(data)}"
            if subscale_mean in subscale_means:
                title_addition += (
                    ", mean by participant: "
                    f"{format_float(subscale_means[subscale_name])}"
                )
            title_addition += ")"
            create_radar_chart(
                survey=survey,
                data=data,
                subscale=subscale_name,
                columns=subscale_columns,
                study_groups=study_groups,
                zeros_are_na=True,
                title_addition=title_addition,
            )
        question_columns = subscale_columns
        if subscale_name in NO_SPIDER_CHART_COLUMNS:
            question_columns += NO_SPIDER_CHART_COLUMNS[subscale_name]
        plot_per_question(
            survey=survey,
            columns=question_columns,
            subscale_name=subscale_name,
        )
