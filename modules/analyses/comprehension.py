"""Code to analyze comprehension results."""

import logging

from pandas import DataFrame

from modules.definitions.constants import (
    ALL_COMPREHENSION_QUESTIONS_NUMBER,
    COMPREHENSION_SCORE_FILE,
    FILE_NAME_ADDITION_PLACEHOLDER,
    PARTICIPANT_ID,
    PARTICIPANT_SCORE_COLUMNS,
    SCORE_COLUMN,
    TIME_POINT_PLACEHOLDER,
    WRONG_ANSWERS_COLUMNS,
)
from modules.definitions.types import Survey, TimePoint, format_time_point_name
from modules.survey_results.get_data import (
    filter_results_by_time_point,
    get_survey_results,
)
from modules.survey_results.redcap_data import get_study_group
from modules.utils.comparisons import (
    PlotData,
    create_comparison_score_plot,
)
from modules.utils.data import write_data_frame
from modules.utils.statistics import (
    NON_INFERIORITY_ALPHA,
    test_non_inferiority_between_study_groups,
)


def get_comprehension_scores_file_path(
    time_point: TimePoint,
    file_name_addition: str = "",
) -> str:
    """Get file path of comprehension score file."""
    return f"{COMPREHENSION_SCORE_FILE}".replace(
        FILE_NAME_ADDITION_PLACEHOLDER,
        file_name_addition,
    ).replace(
        TIME_POINT_PLACEHOLDER,
        time_point.name.lower(),
    )


def get_self_reported_column() -> str:
    """Get the self-reported column from the comprehension data."""
    comprehension_data = get_survey_results(Survey.COMPREHENSION)
    return comprehension_data.columns[3]


def analyze_comprehension(
    time_point: TimePoint,
    drop_questions: list[str],
    file_name_addition: str = "",
) -> tuple[DataFrame, PlotData]:
    """Analyze comprehension results print plot."""
    logger = logging.getLogger(__name__)
    comprehension_data = filter_results_by_time_point(
        Survey.COMPREHENSION,
        time_point,
    )
    participant_scores = []
    wrong_answers = []
    for _, participant_data in comprehension_data.iterrows():
        participant_id = participant_data[PARTICIPANT_ID]
        study_group = get_study_group(participant_id)
        if study_group is None:
            logger.warning(
                "No study group for participant %(participant_id)s",
                {"participant_id": participant_id},
            )
            continue
        participant_results = participant_data.tail(
            ALL_COMPREHENSION_QUESTIONS_NUMBER,
        )
        participant_results = participant_results.drop(
            drop_questions,
        )
        participant_score = len(
            participant_results[participant_results == True],  # noqa: E712
        )
        corrected_comprehension_questions_number = (
            ALL_COMPREHENSION_QUESTIONS_NUMBER - len(drop_questions)
        )
        if participant_score < corrected_comprehension_questions_number:
            for question, answer in participant_results.items():
                if answer != True:  # noqa: E712
                    wrong_answers.append([participant_id, question, answer])
        participant_scores.append(
            [participant_id, participant_score],
        )
    participant_scores = DataFrame(
        participant_scores,
        columns=PARTICIPANT_SCORE_COLUMNS,
    )
    write_data_frame(
        participant_scores,
        get_comprehension_scores_file_path(time_point, file_name_addition),
    )
    pvalue = test_non_inferiority_between_study_groups(
        participant_scores,
        SCORE_COLUMN,
        0.1,
    )
    result = f"[{format_time_point_name(time_point.name).capitalize()}]: "
    result += (
        "Non-inferiority assumed"
        if pvalue < NON_INFERIORITY_ALPHA
        else "Cannot assume non-inferiority"
    )
    result += (
        f",\np = {round(pvalue, ndigits=5)} with ⍺ = {NON_INFERIORITY_ALPHA}"  # noqa: RUF001
        " and Δ = 10%"
    )
    plot_data = create_comparison_score_plot(
        result,
        participant_scores,
        f"comprehension_non_inferiority{file_name_addition}_{time_point.name.lower()}",
        dry_run=False,
        assess_difference=False,
    )

    return DataFrame(
        wrong_answers,
        columns=WRONG_ANSWERS_COLUMNS,
    ), plot_data
