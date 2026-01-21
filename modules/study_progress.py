"""Code to display study progress."""

import pandas as pd

from modules.definitions.constants import (
    PARTICIPANT_ID,
    PROGRESS_DATA_FILE,
    REDCAP_STUDY_GROUPS,
)
from modules.definitions.types import (
    BASELINE_SURVEYS,
    StudyGroup,
    TimePoint,
    format_time_point_name,
)
from modules.survey_results.redcap_data import get_redcap_data, get_study_group
from modules.utils.data import load_data_from_file
from modules.utils.redcap import get_redcap_users


def _get_potentially_lost_cases(
    binary_progress_data: pd.DataFrame,
) -> pd.DataFrame:
    potentially_lost_cases = pd.DataFrame(
        columns=[
            PARTICIPANT_ID,
            "study_arm",
            "testing_completed",
        ],
    )
    redcap_data = get_redcap_data()
    for participant_id in binary_progress_data[PARTICIPANT_ID]:
        study_group = get_study_group(participant_id)
        if study_group is None:
            continue
        binary_participant_progress_data = binary_progress_data[
            binary_progress_data[PARTICIPANT_ID] == participant_id
        ]
        participant_redcap_data = redcap_data[
            redcap_data[PARTICIPANT_ID] == participant_id
        ]
        primary_outcome_answered = binary_participant_progress_data[
            "comprehension_t0"
        ].array[0]
        if not primary_outcome_answered:
            potentially_lost_cases.loc[len(potentially_lost_cases.index)] = [
                participant_id,
                study_group.value,
                participant_redcap_data["testing_completed"].notna().array[0],
            ]
    return potentially_lost_cases


def get_study_progress() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Get study progress based on REDCap data.

    Print progress and return potentially lost cases and partial surveys.
    """
    progress_data = load_data_from_file(
        PROGRESS_DATA_FILE,
    )
    redcap_users = get_redcap_users()
    randomized_users = [
        user for user in redcap_users if user["randomization_complete"] == "2"
    ]
    pharme_users = [
        user
        for user in randomized_users
        if REDCAP_STUDY_GROUPS[user["randomization"]] == StudyGroup.PHARME
    ]
    counseling_users = [
        user
        for user in randomized_users
        if REDCAP_STUDY_GROUPS[user["randomization"]] == StudyGroup.COUNSELING
    ]
    progress_message = f"Participants (baseline): {len(progress_data)}\n"
    progress_message += (
        f"Randomized (data available): {len(randomized_users)} "
        f"({len(pharme_users)} PharMe, {len(counseling_users)} counseling)\n"
    )
    binary_columns = progress_data.columns.difference(
        [PARTICIPANT_ID],
    )
    binary_progress_data = progress_data.copy(deep=True)
    binary_progress_data[binary_columns] = binary_progress_data[
        binary_columns
    ].notna()
    time_point_survey_columns = {}
    time_point_survey_columns["baseline"] = [
        survey.value.name for survey in BASELINE_SURVEYS
    ]
    for time_point in TimePoint:
        survey_columns = [
            column
            for column in binary_progress_data.columns
            if column.endswith(time_point.value.postfix)
        ]
        time_point_survey_columns[time_point.name] = survey_columns
    partial_survey_overview = {}
    for time_point_name, survey_columns in time_point_survey_columns.items():
        time_point_started = 0
        time_point_finished = 0
        formatted_time_point_name = format_time_point_name(
            time_point_name,
        )
        partial_survey_overview[formatted_time_point_name] = pd.DataFrame(
            columns=[PARTICIPANT_ID, *survey_columns],
        )
        for participant_id in binary_progress_data[PARTICIPANT_ID]:
            participant_time_point_data = binary_progress_data[
                binary_progress_data[PARTICIPANT_ID] == participant_id
            ][survey_columns]
            answered_surveys = participant_time_point_data.squeeze().sum()
            if answered_surveys > 0:
                time_point_started += 1
            total_time_point_surveys = len(survey_columns)
            study_group = get_study_group(participant_id)
            # If the participant is in the counseling group and the time point
            # is result return, subtract one survey because of uMARS
            ignore_umars = (
                study_group is not None
                and study_group.name == StudyGroup.COUNSELING.name
                and time_point_name == TimePoint.RESULT_RETURN.name
            )
            if ignore_umars:
                total_time_point_surveys -= 1
            if answered_surveys == total_time_point_surveys:
                time_point_finished += 1
            if (
                answered_surveys > 0
                and answered_surveys != total_time_point_surveys
            ):
                partial_participant_surveys = (
                    participant_time_point_data.copy(
                        deep=True,
                    )
                    .replace(True, "x")  # noqa: FBT003
                    .replace(False, "")  # noqa: FBT003
                )
                if ignore_umars:
                    partial_participant_surveys["u-mars_t0"] = "-"
                partial_participant_surveys[PARTICIPANT_ID] = participant_id
                partial_survey_overview[formatted_time_point_name] = pd.concat(
                    [
                        partial_survey_overview[formatted_time_point_name],
                        partial_participant_surveys,
                    ],
                )
        progress_message += (
            f"{formatted_time_point_name.capitalize()} surveys:"
            f" {time_point_finished}"
        )
        partial_surveys = time_point_started - time_point_finished
        if partial_surveys > 0:
            progress_message += f" (plus {partial_surveys} partial)"
        progress_message += "\n"

    return (
        progress_message,
        _get_potentially_lost_cases(binary_progress_data),
        partial_survey_overview,
    )
