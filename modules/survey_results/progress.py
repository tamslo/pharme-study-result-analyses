"""Code to process progress data."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from modules.definitions.constants import (
    BASELINE_SURVEY_PROGRESS_FILE,
    COUNSELING_GROUP_SURVEY_PROGRESS_FILE,
    PARTICIPANT_ID,
    PHARME_GROUP_SURVEY_PROGRESS_FILE,
    PREPROCESSED_FILE_SUFFIX,
    PROGRESS_CLEANING_LOG,
    PROGRESS_DATA_FILE,
    SURVEY_DIRECTORY,
    SURVEY_TYPE_SEPARATOR,
    TIME_POINT,
)
from modules.definitions.types import ThisShouldNeverHappenError
from modules.survey_results.get_data import get_manual_progress_data
from modules.utils.data import (
    get_last_file_modification,
    get_latest_file_modification,
    load_data_from_file,
    value_is_nan,
    write_data_frame,
)


def _normalize_survey_name(column_name: str) -> str:
    if SURVEY_TYPE_SEPARATOR in column_name:
        column_name = column_name.split(SURVEY_TYPE_SEPARATOR)[1]
    return column_name.replace("-app", "").replace("-counseling", "")


def _normalize_survey_names(progress_data: DataFrame) -> DataFrame:
    progress_data.columns = [
        _normalize_survey_name(column_name)
        for column_name in progress_data.columns
    ]
    return progress_data


def _combine_progress_data() -> DataFrame:
    baseline_progress_data = _normalize_survey_names(
        load_data_from_file(BASELINE_SURVEY_PROGRESS_FILE),
    )
    in_study_progress_data = pd.concat(
        [
            _normalize_survey_names(
                load_data_from_file(PHARME_GROUP_SURVEY_PROGRESS_FILE),
            ),
            _normalize_survey_names(
                load_data_from_file(COUNSELING_GROUP_SURVEY_PROGRESS_FILE),
            ),
        ],
        ignore_index=True,
    )
    combined_data = baseline_progress_data.merge(
        in_study_progress_data,
        how="outer",
        on=PARTICIPANT_ID,
    ).drop_duplicates()
    manual_time_point_overwrites = get_manual_progress_data()
    for participant_id in manual_time_point_overwrites:
        manual_time_points = manual_time_point_overwrites[participant_id]
        for progress_column in manual_time_points:
            time_point = manual_time_points[progress_column]
            row_index = combined_data[
                combined_data[PARTICIPANT_ID] == participant_id
            ].index
            if len(row_index) != 1:
                error_message = (
                    "🚨 Expecting to find exactly one row in "
                    f"progress data for participant ID {participant_id}"
                )
                raise ThisShouldNeverHappenError(error_message)
            combined_data.loc[row_index, progress_column] = time_point
    return combined_data


def _normalize_time_point(time_point: str) -> str:
    return time_point.split(" ")[0]


def _test_that_valid_time_points_are_unique(
    participant_id: str,
    survey_name: str,
    participant_survey_data: DataFrame,
    valid_time_points: list[str],
) -> tuple[list, list]:
    log_content = []
    invalid_time_point_indices = []
    for time_point in valid_time_points:
        surveys_with_time_point = participant_survey_data[
            participant_survey_data[TIME_POINT] == time_point
        ]
        if len(surveys_with_time_point) == 0:
            last_survey_modification = datetime.fromtimestamp(  # noqa: DTZ006
                get_latest_file_modification(
                    Path(SURVEY_DIRECTORY).iterdir(),
                ),
            )
            missing_time_point = datetime.fromisoformat(time_point)
            if last_survey_modification.date() > missing_time_point.date():
                message = (
                    f"⚠️ Time point {time_point} not found for participant "
                    f"{participant_id} in {survey_name}\n"
                )
                log_content.append(message)
        if len(surveys_with_time_point) > 1:
            message = (
                f"ℹ️ Removed duplicate time point {time_point} for "  # noqa: RUF001
                f"participant {participant_id} from {survey_name}\n"
            )
            log_content.append(message)
            survey_content = participant_survey_data.drop(
                TIME_POINT,
                axis=1,
            )
            time_point_content = survey_content.loc[
                surveys_with_time_point.index
            ]
            duplicated_rows = time_point_content.duplicated(
                keep=False,
            ).sum()
            if duplicated_rows != len(surveys_with_time_point):
                message = "⚠️ Rows are not the same! Only keeping last entry.\n"
                log_content.append(message)
            for index, _ in surveys_with_time_point.iloc[:-1].iterrows():
                invalid_time_point_indices.append(index)
    return log_content, invalid_time_point_indices


def _test_that_time_points_are_valid(
    participant_id: str,
    survey_name: str,
    participant_survey_data: DataFrame,
    valid_time_points: list[str],
) -> tuple[list, list]:
    log_content = []
    invalid_time_point_indices = []
    for index, row in participant_survey_data.iterrows():
        time_point = row[TIME_POINT]
        if time_point not in valid_time_points:
            message = (
                f"ℹ️ Removed time point {time_point} not in progress data"  # noqa: RUF001
                f" for participant {participant_id} from {survey_name}\n"
            )
            log_content.append(message)
            invalid_time_point_indices.append(index)
    return log_content, invalid_time_point_indices


def _test_time_point_validity(
    participant_id: str,
    survey_name: str,
    participant_survey_data: DataFrame,
    valid_time_points: list[str],
) -> tuple[list, list]:
    log_content = []
    invalid_time_point_indices = []
    tests = [
        _test_that_time_points_are_valid,
        _test_that_valid_time_points_are_unique,
    ]
    for test in tests:
        new_log_content, new_invalid_indices = test(
            participant_id,
            survey_name,
            participant_survey_data,
            valid_time_points,
        )
        log_content += new_log_content
        invalid_time_point_indices += new_invalid_indices
    return log_content, invalid_time_point_indices


def _get_invalid_time_point_indices(
    survey_name: str,
    survey_data: DataFrame,
    survey_progress_data: DataFrame,
    participant_ids: list[str],
) -> tuple[list, list]:
    invalid_time_point_indices = []
    log_content = []
    for participant_id in participant_ids:
        if participant_id not in survey_data[PARTICIPANT_ID].array:
            continue
        valid_time_points = []
        for column in survey_progress_data.columns:
            if column == PARTICIPANT_ID:
                continue
            participant_progress_data = survey_progress_data[
                survey_progress_data[PARTICIPANT_ID] == participant_id
            ]
            if len(participant_progress_data) == 0:
                continue
            if len(participant_progress_data) > 1:
                error_message = (
                    "🚨 Participant progress should be unique in progress data "
                    f"for {participant_id}!"
                )
                raise ThisShouldNeverHappenError(error_message)
            time_point = participant_progress_data[column].array[0]
            if not value_is_nan(time_point):
                valid_time_points.append(
                    _normalize_time_point(time_point),
                )
        participant_survey_data = survey_data[
            survey_data[PARTICIPANT_ID] == participant_id
        ].copy(deep=True)
        participant_survey_data[TIME_POINT] = participant_survey_data[
            TIME_POINT
        ].apply(_normalize_time_point)
        new_log_content, new_invalid_indices = _test_time_point_validity(
            participant_id,
            survey_name,
            participant_survey_data,
            valid_time_points,
        )
        log_content += new_log_content
        invalid_time_point_indices += new_invalid_indices
    return invalid_time_point_indices, log_content


def maybe_combine_progress_data_and_clean_surveys(
    participant_ids: list[str],
    survey_files_did_change: bool,  # noqa: FBT001
) -> None:
    """Combine progress files if needed."""
    logger = logging.getLogger(__name__)
    progress_files = [
        BASELINE_SURVEY_PROGRESS_FILE,
        PHARME_GROUP_SURVEY_PROGRESS_FILE,
        COUNSELING_GROUP_SURVEY_PROGRESS_FILE,
    ]
    original_data_date = get_latest_file_modification(
        [
            Path(string_path.replace(PREPROCESSED_FILE_SUFFIX, ""))
            for string_path in progress_files
        ],
    )
    preprocessed_data_path = Path(PROGRESS_DATA_FILE)
    if preprocessed_data_path.exists():
        preprocessed_data_date = get_last_file_modification(
            preprocessed_data_path,
        )
        if original_data_date < preprocessed_data_date:
            logger.info("Progress data corresponds to progress files")
            return
    logger.info("Combining progress data and cleaning surveys...")
    progress_data = _combine_progress_data()
    write_data_frame(
        progress_data,
        preprocessed_data_path,
    )
    if survey_files_did_change:
        removed_entries_log_content = []
        for survey_path in sorted(Path(SURVEY_DIRECTORY).iterdir()):
            if PREPROCESSED_FILE_SUFFIX in survey_path.suffixes:
                survey_name = survey_path.name.split(".")[0]
                survey_progress_columns = [
                    column
                    for column in progress_data.columns
                    if column.startswith(_normalize_survey_name(survey_name))
                    and not column.startswith(f"{survey_name}-")
                ]
                if len(survey_progress_columns) == 0:
                    continue
                survey_data = load_data_from_file(survey_path)
                invalid_time_point_indices, log_content = (
                    _get_invalid_time_point_indices(
                        survey_name,
                        survey_data,
                        progress_data[
                            [PARTICIPANT_ID, *survey_progress_columns]
                        ],
                        participant_ids,
                    )
                )
                removed_entries_log_content += log_content
                clean_survey_data = survey_data.drop(invalid_time_point_indices)
                write_data_frame(
                    clean_survey_data,
                    survey_path,
                )
        with Path.open(PROGRESS_CLEANING_LOG, "w") as log_file:
            log_file.writelines(removed_entries_log_content)
