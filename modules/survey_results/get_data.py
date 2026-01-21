"""Code for analysis scripts to get survey data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pandas import DataFrame

from modules.definitions.constants import (
    MANUAL_PROGRESS_DATA,
    PARTICIPANT_ID,
    PROGRESS_DATA_FILE,
    SCORE_COLUMN,
    STUDY_GROUP,
    TIME_POINT,
)
from modules.definitions.types import (
    StudyGroup,
    Survey,
    ThisShouldNeverHappenError,
    TimePoint,
    format_time_point_name,
)
from modules.survey_results.redcap_data import get_redcap_data
from modules.utils.data import (
    get_data_path,
    get_definition_data_path,
    load_answer_definitions,
    load_data_from_file,
    value_is_nan,
)


def get_survey_results(survey: Survey) -> DataFrame:
    """Load survey results from file."""
    survey_path = get_data_path(survey)
    return load_data_from_file(survey_path)


def filter_results_by_study_group(
    survey_data: DataFrame,
    study_group: StudyGroup,
) -> DataFrame:
    """Filter results by one study group."""
    redcap_data = get_redcap_data()
    study_group_participants = redcap_data[
        redcap_data[STUDY_GROUP] == study_group.value
    ][PARTICIPANT_ID].array
    return survey_data[
        survey_data[PARTICIPANT_ID].isin(study_group_participants)
    ]


def _participant_completed_survey(
    participant_id: str,
    survey: Survey,
    time_point: TimePoint,
) -> bool:
    progress_data = load_data_from_file(
        PROGRESS_DATA_FILE,
    )
    survey_progress_column = f"{survey.value.name}_{time_point.value.postfix}"
    progress_time_point = progress_data[
        progress_data[PARTICIPANT_ID] == participant_id
    ][survey_progress_column].array
    if len(progress_time_point) != 1:
        error_message = (
            "🚨 The selected time point should be unique per "
            "participant and time point!"
        )
        raise ThisShouldNeverHappenError(error_message)
    progress_time_point = progress_time_point[0]
    return not value_is_nan(progress_time_point)


def filter_results_by_time_point(
    survey: Survey,
    time_point: TimePoint,
) -> DataFrame:
    """Filter survey data by specific time point."""
    survey_results = get_survey_results(survey)
    time_point_data = DataFrame(
        columns=survey_results.columns,
    )
    current_row = 0
    for participant_id in survey_results[PARTICIPANT_ID].unique():
        participant_data = survey_results[
            survey_results[PARTICIPANT_ID] == participant_id
        ].sort_values(TIME_POINT)
        time_point_missing = not _participant_completed_survey(
            participant_id,
            survey,
            time_point,
        )
        if time_point_missing:
            continue
        time_point_index = time_point.value.index
        for other_time_point in TimePoint:
            if other_time_point.value.index == time_point.value.index:
                break
            if not _participant_completed_survey(
                participant_id,
                survey,
                other_time_point,
            ):
                time_point_index -= 1
        if time_point_index > len(participant_data.index) - 1:
            logger = logging.getLogger(__name__)
            logger.warning(
                "⚠️ Cannot select %(survey_name)s data for %(time_point)s of "
                "participant %(participant_id)s because the survey data was "
                "not loaded yet",
                {
                    "survey_name": survey.value.name,
                    "time_point": format_time_point_name(time_point.name),
                    "participant_id": participant_id,
                },
            )
            continue
        participant_time_point_data = participant_data.iloc[time_point_index]
        time_point_data.loc[current_row] = participant_time_point_data
        current_row += 1
    return time_point_data


class UndefinedScoresError(Exception):
    """Error returned if scores are not defined."""


def get_single_score(
    survey: Survey,
    question_title: str,
    answer: any,
) -> int | None:
    """Get the answer score for a survey question."""
    answer_definitions = load_answer_definitions(
        survey,
        question_title,
    )
    answer_definition = next(
        (
            answer_definition
            for answer_definition in answer_definitions
            if answer_definition["key"] == answer
        ),
        None,
    )
    if answer_definition is None:
        return None
    if "score" in answer_definition:
        answer_score = answer_definition["score"]
    elif survey is Survey.COMPREHENSION:
        comprehension_scores = {
            "strongly_disagree": 1,
            "disagree": 2,
            "agree": 3,
            "strongly_agree": 4,
        }
        answer_score = comprehension_scores[answer_definition["key"]]
    else:
        message = f"Define scores for {survey.name}"
        raise UndefinedScoresError(message)
    return answer_score


def get_defined_scores(survey: Survey, data: DataFrame = None) -> DataFrame:
    """Get scores per participant for a survey.

    Scores are based on scores defined in the survey definition.
    """
    if data is None:
        data = get_survey_results(survey)
    definitions = load_data_from_file(
        get_definition_data_path(survey),
    )
    score_data = []
    for _, row in data.iterrows():
        participant_id = row[PARTICIPANT_ID]
        score = 0
        all_not_answered = True
        for _, question_row in definitions.iterrows():
            question_title = question_row["title"]
            if question_title not in row:
                continue
            answer = row[question_title]
            answer_score = get_single_score(survey, question_title, answer)
            if answer_score is None:
                continue
            all_not_answered = False
            score += answer_score
        if all_not_answered:
            score_data.append([participant_id, None])
        else:
            score_data.append(
                [participant_id, score],
            )
    return DataFrame(score_data, columns=[PARTICIPANT_ID, SCORE_COLUMN])


def get_manual_progress_data() -> dict:
    """Get manual progress data."""
    file_path = Path(MANUAL_PROGRESS_DATA)
    if not file_path.exists():
        return {}
    with Path.open(MANUAL_PROGRESS_DATA, "r") as manual_progress_data_file:
        return json.load(manual_progress_data_file)
