"""Code to create demographic table."""

import logging
from collections import OrderedDict

import pandas as pd
from pandas import DataFrame, Series

from modules.analyses.health_literacy import (
    HEALTH_LITERACY_COLUMN,
    HEALTH_LITERACY_LABELS,
    HEALTH_LITERACY_TITLE,
    get_health_literacy_scores,
)
from modules.definitions.constants import (
    MULTIPLE_VALUES_SEPARATOR,
    PARTICIPANT_ID,
    SCORE_COLUMN,
)
from modules.definitions.types import StudyGroup, Survey
from modules.survey_results.get_data import (
    filter_results_by_study_group,
    get_defined_scores,
    get_survey_results,
)
from modules.survey_results.redcap_data import redcap_data_are_complete
from modules.utils.data import get_label_definition
from modules.utils.output_formatting import (
    format_float,
    format_output_label,
    format_percentage,
)
from modules.utils.sorting import sort_by_label
from modules.utils.statistics import (
    are_study_groups_different_categorical,
    are_study_groups_different_parametric,
)


class Demographic:
    """Robust definition for displaying a single demographic."""

    def __init__(self, name: str, column: str) -> None:
        """Initialize demographic.

        Sets the display name, the column number in the table, and how to format
        values.
        """
        self.name = name
        self.column = column


RACE_COLUMN = "What is your race?"

SELF_EFFICACY_TITLE = "General self efficacy**"
SELF_EFFICACY_FOOTNOTE = "** Scores from 10 (very low) to 40 (very high)"

BASELINE_KNOWLEDGE_TITLE = "Baseline knowledge***"
BASELINE_KNOWLEDGE_FOOTNOTE = "*** Scores from 0 (low) to 5 (high)"

demographics = [
    Demographic("age", "What is your age?"),
    Demographic(
        "gender",
        "What gender do you identify as?",
    ),
    Demographic(
        "ethnicity",
        "Please specify your ethnicity.",
    ),
    Demographic("race", RACE_COLUMN),
    Demographic(
        "highest education",
        "What is your highest level of education?",
    ),
]


def _get_descriptive_stats(data: DataFrame, column: str) -> Series:
    return Series(
        [
            f"{data[column].min()}-{data[column].max()}",
            format_float(data[column].mean()),
        ],
        index=["Range", "Mean"],
    )


def _get_counts(
    data: DataFrame,
    column: str,
    label_definition: OrderedDict,
    study_group: StudyGroup = None,
) -> Series:
    if study_group is not None:
        participant_data = filter_results_by_study_group(
            data,
            study_group,
        )
    else:
        participant_data = data
    count_data = participant_data[column]
    return count_data.value_counts().sort_index(
        key=lambda values: sort_by_label(values, label_definition),
    )


def _format_demographic_count(counts: Series, value: str) -> str:
    if value not in counts:
        return "0"
    count = counts[value]
    total = counts.to_numpy().sum()
    return f"{count} ({format_percentage(count / total)}%)"


def _get_count_rows(
    title: str,
    data: DataFrame,
    column: str,
    label_definition: OrderedDict,
) -> list[list]:
    table_rows = []
    all_counts = _get_counts(
        data,
        column,
        label_definition,
    )
    pharme_counts = _get_counts(
        data,
        column,
        label_definition,
        StudyGroup.PHARME,
    )
    counseling_counts = _get_counts(
        data,
        column,
        label_definition,
        StudyGroup.COUNSELING,
    )
    comparison_result = are_study_groups_different_categorical(data, column)
    for value in all_counts.keys():  # noqa: SIM118
        demographic_row = [
            title,
            format_float(comparison_result.p_value),
            format_output_label(value, label_definition),
            _format_demographic_count(all_counts, value),
            _format_demographic_count(counseling_counts, value),
            _format_demographic_count(pharme_counts, value),
        ]
        table_rows.append(demographic_row)
    return table_rows


def _get_race_label_definition() -> OrderedDict:
    label_definition = get_label_definition(
        Survey.DEMOGRAPHICS,
        RACE_COLUMN,
    )
    label_definition.update({"mixed": "Mixed race*"})
    label_definition.move_to_end("unknown")
    return label_definition


def _get_demographic_rows(
    demographics_data: DataFrame,
    demographic: Demographic,
) -> list[list]:
    label_definition = (
        get_label_definition(Survey.DEMOGRAPHICS, demographic.column)
        if demographic.column is not RACE_COLUMN
        else _get_race_label_definition()
    )
    return _get_count_rows(
        demographic.name.capitalize(),
        demographics_data,
        demographic.column,
        label_definition,
    )


def _get_health_literacy_rows() -> DataFrame:
    return _get_count_rows(
        HEALTH_LITERACY_TITLE,
        get_health_literacy_scores(),
        HEALTH_LITERACY_COLUMN,
        HEALTH_LITERACY_LABELS,
    )


def _get_descriptive_rows(title: str, data: DataFrame, column: str) -> Series:
    table_rows = []
    all_stats = _get_descriptive_stats(data, column)
    pharme_data = filter_results_by_study_group(
        data,
        StudyGroup.PHARME,
    )
    pharme_stats = _get_descriptive_stats(
        pharme_data,
        column,
    )
    counseling_data = filter_results_by_study_group(
        data,
        StudyGroup.COUNSELING,
    )
    counseling_stats = _get_descriptive_stats(
        counseling_data,
        column,
    )
    comparison_result = are_study_groups_different_parametric(data, column)
    for value in all_stats.keys():  # noqa: SIM118
        demographic_row = [
            title,
            format_float(comparison_result.p_value),
            value,
            all_stats[value],
            pharme_stats[value],
            counseling_stats[value],
        ]
        table_rows.append(demographic_row)
    return table_rows


def _get_self_efficacy_rows() -> DataFrame:
    return _get_descriptive_rows(
        SELF_EFFICACY_TITLE,
        get_defined_scores(Survey.GENERAL_SELF_EFFICACY),
        SCORE_COLUMN,
    )


def _get_baseline_knowledge_rows() -> DataFrame:
    return _get_descriptive_rows(
        BASELINE_KNOWLEDGE_TITLE,
        get_defined_scores(Survey.BASELINE_KNOWLEDGE),
        SCORE_COLUMN,
    )


def _get_participant_count(data: DataFrame) -> int:
    return len(data[PARTICIPANT_ID].unique())


def _merge_multiple_race(demographics_data: DataFrame) -> tuple[DataFrame, str]:
    multiple_races = {
        value
        for value in demographics_data[RACE_COLUMN]
        if MULTIPLE_VALUES_SEPARATOR in str(value)
    }
    multiple_race_description = "*"
    label_definition = _get_race_label_definition()
    for race in sorted(
        multiple_races,
        key=lambda values: sort_by_label(values, label_definition),
    ):
        prefix = " " if multiple_race_description == "*" else "; "
        count = demographics_data[demographics_data[RACE_COLUMN] == race][
            RACE_COLUMN
        ].count()
        multiple_race_description += (
            f"{prefix}{format_output_label(race, label_definition)} ({count})"
        )
    demographics_data[RACE_COLUMN] = demographics_data[RACE_COLUMN].apply(
        lambda value: "mixed" if value in multiple_races else value,
    )
    return demographics_data, multiple_race_description


def get_demographic_table() -> tuple[DataFrame, list[str]]:
    """Create the demographic table."""
    demographics_data, multiple_race_description = _merge_multiple_race(
        get_survey_results(
            Survey.DEMOGRAPHICS,
        ),
    )
    if not redcap_data_are_complete():
        logger = logging.getLogger(__name__)
        logger.info("⚠️ Not all participants have study groups assigned yet!")
    demographics_table_data = []
    for demographic in demographics:
        demographics_table_data += _get_demographic_rows(
            demographics_data,
            demographic,
        )
    demographics_table_data += _get_health_literacy_rows()
    demographics_table_data += _get_self_efficacy_rows()
    demographics_table_data += _get_baseline_knowledge_rows()

    total_count = _get_participant_count(demographics_data)
    counseling_count = _get_participant_count(
        filter_results_by_study_group(demographics_data, StudyGroup.COUNSELING),
    )
    pharme_count = _get_participant_count(
        filter_results_by_study_group(demographics_data, StudyGroup.PHARME),
    )

    demographics_table = DataFrame(
        demographics_table_data,
        columns=pd.MultiIndex.from_tuples(
            [
                ("Demographic", ""),
                ("Groups different (p)", ""),
                ("Value", ""),
                ("Count (%)", f"Total (n = {total_count})"),
                ("Count (%)", f"Counseling group (n = {counseling_count})"),
                ("Count (%)", f"PharMe group (n = {pharme_count})"),
            ],
        ),
    )
    return demographics_table.set_index(
        ["Demographic", "Groups different (p)", "Value"],
    ), [
        multiple_race_description,
        SELF_EFFICACY_FOOTNOTE,
        BASELINE_KNOWLEDGE_FOOTNOTE,
    ]
