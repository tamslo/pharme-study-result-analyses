"""Data utils."""

from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame, Series

from modules.definitions.constants import (
    DATA_FILE_SUFFIX,
    MANUAL_FILE_SUFFIX,
    PREPROCESSED_FILE_SUFFIX,
    PROGRESS_DATA_FILE,
    SURVEY_DEFINITION_DIRECTORY,
    SURVEY_DIRECTORY,
    SURVEY_TYPE_SEPARATOR,
)

if TYPE_CHECKING:
    from modules.definitions.types import Survey


def get_latest_file_modification(paths: list[Path]) -> float:
    """Get latest modification Unix time stamp from a list of files."""
    last_modified = datetime.now().timestamp()  # noqa: DTZ005
    for file_path in paths:
        if PREPROCESSED_FILE_SUFFIX in file_path.suffixes:
            continue
        if file_path.name.startswith("."):
            continue
        file_last_modified = get_last_file_modification(file_path)
        last_modified = max(last_modified, file_last_modified)
    return last_modified


def get_last_file_modification(path: Path) -> float:
    """Get the Unix time stamp of the last file modification."""
    return path.stat().st_mtime


def load_data_from_file(path: Path) -> DataFrame:
    """Load DataFrame from CSV file path."""
    return pd.read_csv(path)


def get_data_path(data: Survey, preprocessed: bool = True) -> Path:  # noqa: FBT001, FBT002
    """Get the (preprocessed) data path for a survey."""
    middle_suffix = PREPROCESSED_FILE_SUFFIX if preprocessed else ""
    file_name = f"{data.value.name}{middle_suffix}{DATA_FILE_SUFFIX}"
    return Path(f"{SURVEY_DIRECTORY}/{file_name}")


def get_manual_path(unprocessed_path: Path) -> Path:
    """Get the manual path from an unprocessed path."""
    return unprocessed_path.with_suffix(
        f"{MANUAL_FILE_SUFFIX}{DATA_FILE_SUFFIX}",
    )


def write_data_frame(
    data_frame: DataFrame,
    path: Path | str,
    mode: str = "w",
) -> None:
    """Write data frame to CSV with default settings."""
    data_frame.to_csv(
        path,
        index=False,
        mode=mode,
    )


def value_is_nan(value) -> bool:  # noqa: ANN001
    """Test if a (non-numeric) value is NaN."""
    return value != value  # noqa: PLR0124


def get_definition_data_path(survey: Survey) -> Path:
    """Get the definition data path for a survey."""
    return Path(
        f"{SURVEY_DEFINITION_DIRECTORY}/{survey.value.type.name.lower()}"
        f"{SURVEY_TYPE_SEPARATOR}{survey.value.name}{DATA_FILE_SUFFIX}",
    )


def _get_answer_definition(
    survey: Survey,
    row_index: int | str,
) -> Series:
    definitions = load_data_from_file(get_definition_data_path(survey))
    if type(row_index) is str:
        row_index = definitions[definitions["title"] == row_index].index[0]
    return definitions.iloc[row_index]


def load_answer_definitions(
    survey: Survey,
    row_index: int | str,
) -> list | None:
    """Get answer definitions for a question."""
    answer_definition = _get_answer_definition(survey, row_index)
    answer_type = answer_definition["type"]
    if answer_type == "YESNO_CHOICE":
        return [
            {"key": "yes", "label": "Yes"},
            {"key": "no", "label": "No"},
        ]
    label_definition_string = answer_definition["options"]
    if value_is_nan(label_definition_string):
        return None
    label_definition_json = (
        label_definition_string.replace(
            " '",
            ' "',
        )
        .replace(
            "',",
            '",',
        )
        .replace(
            "{'",
            '{"',
        )
        .replace(
            "'}",
            '"}',
        )
        .replace(
            "':",
            '":',
        )
        .replace(
            '""',
            '"',
        )
    )
    return json.loads(label_definition_json)


def get_label_definition(
    survey: Survey,
    row_index: int | str,
    remove_column_formulation: str | None = None,
    column_formulation_replacement: str | None = None,
) -> OrderedDict | None:
    """Get label keys and values from a dictionary file."""
    if remove_column_formulation is not None:
        row_index = row_index.replace(
            remove_column_formulation,
            column_formulation_replacement,
        )
    answer_definitions = load_answer_definitions(survey, row_index)
    if answer_definitions is None:
        return None
    label_definition = OrderedDict()
    for label_item in answer_definitions:
        label_definition[label_item["key"]] = label_item["label"]
    return label_definition


def is_score_answer(survey: Survey, column: str) -> bool:
    """Test if answers are scores."""
    return _get_answer_definition(survey, column)["type"] == "H_SCALE"


class ScoreDefinition:
    """Holds items related to H_SCALE scores."""

    description: str | None
    min: float
    min_label: str
    max: float
    max_label: str

    def __init__(self, answer_definition: Series) -> None:  # noqa: D107
        if "description" in answer_definition and not value_is_nan(
            answer_definition["description"],
        ):
            self.description = answer_definition["description"]
        else:
            self.description = None
        self.min = answer_definition["min"]
        self.min_label = answer_definition["minLabel"]
        self.max = answer_definition["max"]
        self.max_label = answer_definition["maxLabel"]

    def get_info_string(self) -> str:
        """Get information string from score definition, e.g., for a title."""
        return (
            f"{self.min_label} ({int(self.min)!s}) ↔ "
            f"{self.max_label} ({int(self.max)!s})"
        )


def get_score_definition(survey: Survey, column: str) -> ScoreDefinition:
    """Get a ScoreDefinition for a survey and column."""
    answer_definition = _get_answer_definition(survey, column)
    return ScoreDefinition(answer_definition)


def is_free_text_answer(survey: Survey, column: str) -> bool:
    """Test if answer is a free text answer."""
    return _get_answer_definition(survey, column)["type"] == "TEXTAREA"


def get_score_interpretation(
    score: float,
    max_score_definition: OrderedDict,
) -> str:
    """Get the lowest matching score interpretation."""
    for interpretation, max_value in max_score_definition.items():
        if score <= max_value:
            return interpretation
    return None


def has_multiple_time_points(survey: Survey) -> bool:
    """Test if a survey has multiple time points."""
    progress_data = load_data_from_file(
        PROGRESS_DATA_FILE,
    )
    survey_columns = [
        column
        for column in progress_data.columns
        if column == survey.value.name
        or column.startswith(f"{survey.value.name}_")
    ]
    return len(survey_columns) > 1
