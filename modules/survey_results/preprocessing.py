"""Bundles all preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from modules.definitions.constants import (
    DATA_FILE_SUFFIX,
    GENERATED_DATA_FILES,
    MANUAL_FILE_SUFFIX,
    PARTICIPANT_ID,
    PREPROCESSED_FILE_SUFFIX,
    PREPROCESSING_DIRECTORIES,
    TIME_POINT,
)
from modules.survey_results.comprehension import maybe_map_comprehension_data
from modules.survey_results.progress import (
    maybe_combine_progress_data_and_clean_surveys,
)
from modules.survey_results.redcap_data import (
    maybe_update_redcap_data,
)
from modules.survey_results.satisfaction import normalize_satisfaction_surveys
from modules.utils.anonymization import anonymize_results
from modules.utils.data import (
    get_last_file_modification,
    get_manual_path,
    load_data_from_file,
    write_data_frame,
)


def _should_skip_file(path: Path) -> bool:
    return (
        path.suffix != DATA_FILE_SUFFIX
        or str(path) in GENERATED_DATA_FILES
        or PREPROCESSED_FILE_SUFFIX in path.suffixes
        or MANUAL_FILE_SUFFIX in path.suffixes
        or path.stat().st_size <= 1
    )


def _get_preprocessing_file_list() -> list[Path]:
    preprocessing_file_list = []
    for path_string in PREPROCESSING_DIRECTORIES:
        preprocessing_directory = Path(path_string)
        for file_path in sorted(preprocessing_directory.iterdir()):
            if _should_skip_file(file_path):
                continue
            preprocessing_file_list.append(file_path)
    return preprocessing_file_list


def _load_original_survey_data(file_path: Path) -> DataFrame:
    original_survey_data = load_data_from_file(file_path)
    manual_file_path = get_manual_path(file_path)
    if manual_file_path.exists():
        ehive_data = original_survey_data.copy(deep=True)
        original_survey_data = pd.concat(
            [
                ehive_data,
                load_data_from_file(manual_file_path),
            ],
            ignore_index=True,
        )
    if TIME_POINT in original_survey_data.columns:
        original_survey_data = original_survey_data.sort_values(TIME_POINT)
    return original_survey_data


def _files_did_change() -> bool:
    preprocessing_paths = _get_preprocessing_file_list()
    generated_data_paths = [
        Path(string_path) for string_path in GENERATED_DATA_FILES
    ]
    for file_path in [*preprocessing_paths, *generated_data_paths]:
        if not Path(file_path).exists():
            return True
    for file_path in preprocessing_paths:
        preprocessed_path = get_preprocessed_path(file_path)
        if not preprocessed_path.exists():
            return True
        original_data_date = get_last_file_modification(file_path)
        preprocessed_data_date = get_last_file_modification(preprocessed_path)
        if original_data_date > preprocessed_data_date:
            return True
        manual_path = get_manual_path(file_path)
        if manual_path.exists():
            manual_data_date = get_last_file_modification(manual_path)
            if manual_data_date > preprocessed_data_date:
                return True
    return False


def get_preprocessed_path(unprocessed_path: Path) -> Path:
    """Get the preprocessed path for an unprocessed path."""
    return unprocessed_path.with_suffix(
        f"{PREPROCESSED_FILE_SUFFIX}{DATA_FILE_SUFFIX}",
    )


def maybe_preprocess_study_results() -> None:
    """Potentially do preprocessing, if needed."""
    files_did_change = _files_did_change()
    participant_ids = set()
    for file_path in _get_preprocessing_file_list():
        if files_did_change:
            survey_results = anonymize_results(
                _load_original_survey_data(file_path),
            )
            write_data_frame(
                survey_results,
                get_preprocessed_path(file_path),
            )
        else:
            survey_results = load_data_from_file(
                get_preprocessed_path(file_path),
            )
        participant_ids.update(survey_results[PARTICIPANT_ID].array)
    maybe_update_redcap_data(participant_ids)
    maybe_combine_progress_data_and_clean_surveys(
        participant_ids,
        files_did_change,
    )
    maybe_map_comprehension_data()
    normalize_satisfaction_surveys()
