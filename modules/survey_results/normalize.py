"""Code to normalize app and counseling surveys to display joint results.

The study group can be inferred from the participant ID later anyway.
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from modules.definitions.constants import (
    PARTICIPANT_ID,
    SCORE_COLUMN,
    TIME_POINT,
)
from modules.definitions.types import Survey
from modules.survey_results.get_data import get_survey_results
from modules.utils.data import (
    get_definition_data_path,
    load_data_from_file,
)


def _replace_in_column(
    column: str,
    remove_column_formulations: list[str],
    column_formulation_replacement: str,
) -> str:
    normalized_column = column
    for formulation in remove_column_formulations:
        normalized_column = normalized_column.replace(
            formulation,
            column_formulation_replacement,
        )
    return normalized_column


def replace_in_columns(
    survey: Survey,
    remove_column_formulations: list[str],
    column_formulation_replacement: str,
) -> list[str]:
    """Get columns with replaced formulations."""
    survey_definition = load_data_from_file(
        get_definition_data_path(survey),
    )
    columns = survey_definition.title
    return [
        PARTICIPANT_ID,
        TIME_POINT,
        SCORE_COLUMN,
        *[
            _replace_in_column(
                column,
                remove_column_formulations,
                column_formulation_replacement,
            )
            for column in columns
        ],
    ]


def _get_normalized_survey(
    survey: Survey,
    remove_column_formulations: list[str],
    column_formulation_replacement: str,
) -> DataFrame:
    survey_results = get_survey_results(survey)
    survey_results.columns = replace_in_columns(
        survey,
        remove_column_formulations,
        column_formulation_replacement,
    )
    return survey_results


def get_normalized_survey_data(
    counseling_survey: Survey,
    pharme_survey: Survey,
    normalized_survey: Survey,
    remove_column_formulations: list[str],
    column_formulation_replacement: str,
) -> DataFrame:
    """Normalize app and counseling survey data."""
    _create_normalized_dictionary(
        survey=pharme_survey,
        normalized_survey=normalized_survey,
        remove_column_formulations=remove_column_formulations,
        column_formulation_replacement=column_formulation_replacement,
    )
    return pd.concat(
        [
            _get_normalized_survey(
                counseling_survey,
                remove_column_formulations,
                column_formulation_replacement,
            ),
            _get_normalized_survey(
                pharme_survey,
                remove_column_formulations,
                column_formulation_replacement,
            ),
        ],
        ignore_index=True,
    ).sort_values(TIME_POINT)


def _create_normalized_dictionary(
    survey: Survey,
    normalized_survey: Survey,
    remove_column_formulations: list[str],
    column_formulation_replacement: str,
) -> None:
    """Create normalized dictionary file."""
    with Path.open(
        get_definition_data_path(survey),
        "r",
    ) as survey_definition_file:
        survey_definition = survey_definition_file.read()
        normalized_survey_definition = _replace_in_column(
            column=survey_definition,
            remove_column_formulations=remove_column_formulations,
            column_formulation_replacement=column_formulation_replacement,
        )
        with Path.open(
            get_definition_data_path(normalized_survey),
            "w",
        ) as normalized_definition_file:
            normalized_definition_file.write(normalized_survey_definition)
