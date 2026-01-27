"""Code to analyze correlations."""

import json
from collections.abc import Callable
from pathlib import Path

from pandas import DataFrame, Series

from modules.analyses.app_rating import get_overall_app_rating_data
from modules.analyses.comprehension import get_comprehension_scores_file_path
from modules.analyses.self_efficacy import get_self_efficacy_feelings_columns
from modules.definitions.constants import (
    COMPREHENSION_DATA,
    EHIVE_ID,
    PARTICIPANT_ID,
    SCORE_COLUMN,
)
from modules.definitions.types import Survey, TimePoint, format_time_point_name
from modules.survey_results.get_data import (
    filter_results_by_time_point,
    get_defined_scores,
    get_survey_results,
)
from modules.utils.anonymization import get_participant_id_map
from modules.utils.data import has_multiple_time_points, load_data_from_file
from modules.utils.output_formatting import format_output_label
from modules.utils.redcap import get_pharme_id, get_redcap_users


def _get_correlation_data(
    data: list[DataFrame],
    drop_participant_ids: bool = True,  # noqa: FBT001, FBT002
) -> DataFrame:
    correlation_data = DataFrame(columns=[PARTICIPANT_ID])
    for current_data in data:
        correlation_data = correlation_data.merge(
            current_data,
            on=PARTICIPANT_ID,
            how="outer",
        )
    if drop_participant_ids:
        correlation_data = correlation_data.drop(PARTICIPANT_ID, axis=1)
    return correlation_data


def _get_ordinal_age_data(demographic_data: DataFrame) -> DataFrame:
    age_column = "What is your age?"
    age_data = demographic_data[[PARTICIPANT_ID, "What is your age?"]].copy(
        deep=True,
    )
    levels = sorted(age_data[age_column].unique())
    age_data["age"] = [
        levels.index(age_category) for age_category in age_data[age_column]
    ]
    return age_data.drop(age_column, axis=1)


def _get_score_data(
    survey: Survey,
    result_column: str,
    get_scores: Callable | None = None,
    columns: list[str] | None = None,
) -> DataFrame:
    if has_multiple_time_points(survey):
        score_data = DataFrame(columns=[PARTICIPANT_ID])
        for time_point in TimePoint:
            if get_scores is not None:
                current_scores = get_scores(time_point)
            else:
                data = filter_results_by_time_point(survey, time_point)
                if columns is not None:
                    data = data[[PARTICIPANT_ID, *columns]]
                current_scores = get_defined_scores(
                    survey,
                    data=data,
                )
            current_scores.columns = [
                PARTICIPANT_ID,
                f"{result_column}_({format_time_point_name(time_point.name)})",
            ]
            score_data = score_data.merge(
                current_scores,
                on=PARTICIPANT_ID,
                how="outer",
            )
        return score_data
    score_data = (
        get_scores() if get_scores is not None else get_defined_scores(survey)
    )
    score_data.columns = [PARTICIPANT_ID, result_column]
    return score_data


def _get_umars_data() -> DataFrame:
    overall_rating_data, _, participant_ids = get_overall_app_rating_data()
    umars_data = []
    for index, participant_id in enumerate(participant_ids):
        subscale_means = overall_rating_data[index]
        umars_data.append([participant_id, Series(subscale_means).mean()])
    return DataFrame(umars_data, columns=[PARTICIPANT_ID, SCORE_COLUMN])


def _get_actions_subset_data(
    time_point: TimePoint,
    columns: list[str],
) -> DataFrame:
    survey_data = filter_results_by_time_point(Survey.ACTIONS, time_point)
    subset_data = []
    for _, row in survey_data.iterrows():
        participant_id = row[PARTICIPANT_ID]
        participant_data = any(row[columns] == "yes")
        subset_data.append([participant_id, participant_data])
    return DataFrame(subset_data, columns=[PARTICIPANT_ID, "bool"])


def _get_hcp_communication_data(time_point: TimePoint) -> DataFrame:
    hcp_communication_columns = [
        (
            "Have you shared or do you plan to share your PGx test results "
            "with your primary care doctor or other doctors involved in your "
            "care?"
        ),
        (
            "Have you shared or do you plan to share your test results with "
            "your pharmacist?"
        ),
        (
            "Have you shared or do you plan to share your test results with "
            "other health care providers (i.e., not doctors or pharmacists) "
            "involved in your care?"
        ),
    ]
    return _get_actions_subset_data(time_point, hcp_communication_columns)


def _get_own_medication_change_data(time_point: TimePoint) -> DataFrame:
    return _get_actions_subset_data(
        time_point,
        [
            (
                "Have you changed your over-the-counter medication based"
                " on your PGx test results?"
            ),
        ],
    )


def _get_hcp_medication_change_data(time_point: TimePoint) -> DataFrame:
    return _get_actions_subset_data(
        time_point,
        [
            (
                "Has your doctor changed your medications based on your PGx "
                "test results or discussed plans with you to do so?"
            ),
        ],
    )


def _get_correlation_columns(
    variable: str,
    correlations: DataFrame,
) -> list[str]:
    return (
        [variable]
        if variable in correlations.columns
        else [
            column
            for column in correlations.columns
            if column.startswith(variable)
        ]
    )


def _get_correlation_interpretation(value: float) -> str:
    # Based on "psychology" interpretation from
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC6107969/
    absolute_value = abs(value)
    if absolute_value == 1:
        return "perfect"
    if absolute_value >= 0.7:  # noqa: PLR2004
        return "strong"
    if absolute_value >= 0.4:  # noqa: PLR2004
        return "moderate"
    if absolute_value > 0:
        return "weak"
    return "none"


def _is_time_point_column(variable: str, column: str) -> bool:
    return variable != column


def _get_medication_comprehension_column_name(
    comprehension_column: str,
    time_point: TimePoint,
    medication: str,
) -> str:
    return (
        f"{comprehension_column}_{medication}_"
        f"({format_time_point_name(time_point.name)})"
    )


def _get_comprehension_questionnaire_medications() -> list[str]:
    with Path.open(COMPREHENSION_DATA) as comprehension_data_file:
        comprehension_data = json.load(comprehension_data_file)
    return sorted(
        next(iter(comprehension_data.values()))["medications"].keys(),
    )


def _format_correlation_variable(
    variable: str,
) -> str:
    formatted_variable = format_output_label(variable, label_definition=None)
    return (
        formatted_variable.replace("Hcp", "HCP")
        .replace(
            "Taking medication",
            "Taking",
        )
        .replace(
            "Medication specific comprehension",
            "Medication-specific comprehension for",
        )
    )


def _get_specific_medication_data(
    taking_medication_column: str,
    comprehension_column: str,
) -> DataFrame:
    with Path.open(COMPREHENSION_DATA) as comprehension_data_file:
        comprehension_data = json.load(comprehension_data_file)
    redcap_users = get_redcap_users()
    participant_map = get_participant_id_map()
    specific_medication_data = []
    questionnaire_medications = _get_comprehension_questionnaire_medications()
    taking_medications_columns = [
        f"{taking_medication_column}_{medication}"
        for medication in questionnaire_medications
    ]
    for redcap_user in redcap_users:
        ehive_id = redcap_user[EHIVE_ID]
        pharme_id = get_pharme_id(redcap_users, ehive_id)
        if pharme_id in comprehension_data:
            anonymous_id = participant_map[ehive_id]
            participant_comprehension_data = comprehension_data[pharme_id]
            participant_medication_data = [
                participant_comprehension_data["medications"][medication]
                == "true"
                for medication in questionnaire_medications
            ]
            specific_medication_data.append(
                [
                    anonymous_id,
                    *participant_medication_data,
                ],
            )
    medication_placeholder = "MEDICATION_PLACEHOLDER"
    medication_question = (
        "According to your PGx test result, if you ever needed to take the "
        f"medication {medication_placeholder}, could you take it at standard "
        "dosage?"
    )
    specific_medication_data = DataFrame(
        specific_medication_data,
        columns=[PARTICIPANT_ID, *taking_medications_columns],
    )
    medication_comprehension_data = {
        PARTICIPANT_ID: [],
    }
    comprehension_time_point_data = {}
    participant_ids = []
    for time_point in TimePoint:
        current_comprehension_data = filter_results_by_time_point(
            Survey.COMPREHENSION,
            time_point,
        )
        participant_ids += current_comprehension_data[PARTICIPANT_ID].to_list()
        comprehension_time_point_data[time_point] = current_comprehension_data
        for medication in questionnaire_medications:
            medication_comprehension_data[
                _get_medication_comprehension_column_name(
                    comprehension_column,
                    time_point,
                    medication,
                )
            ] = []
    for participant_id in Series(participant_ids).unique():
        medication_comprehension_data[PARTICIPANT_ID].append(participant_id)
        for time_point in TimePoint:
            time_point_data = comprehension_time_point_data[time_point]
            for medication in questionnaire_medications:
                data_column = medication_question.replace(
                    medication_placeholder,
                    medication,
                )
                participant_data = time_point_data[
                    time_point_data[PARTICIPANT_ID] == participant_id
                ][data_column]
                if len(participant_data) > 1:
                    message = "Participant ID should be unique!"
                    raise Exception(message)  # noqa: TRY002
                if len(participant_data) == 1:
                    participant_medication_result = participant_data.iloc[0]
                else:
                    participant_medication_result = None
                medication_comprehension_data[
                    _get_medication_comprehension_column_name(
                        comprehension_column,
                        time_point,
                        medication,
                    )
                ].append(participant_medication_result)
    medication_comprehension_data = DataFrame.from_dict(
        medication_comprehension_data,
    )
    return _get_correlation_data(
        [specific_medication_data, medication_comprehension_data],
        drop_participant_ids=False,
    )


def analyze_correlations() -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Define questions and analyze correlations."""
    demographic_data = get_survey_results(
        Survey.DEMOGRAPHICS,
    )
    correlation_data = _get_correlation_data(
        [
            _get_ordinal_age_data(demographic_data),
            _get_score_data(
                Survey.HEALTH_LITERACY,
                "health_literacy",
            ),
            _get_score_data(
                Survey.COMPREHENSION,
                "comprehension",
                lambda time_point: load_data_from_file(
                    get_comprehension_scores_file_path(time_point),
                ),
            ),
            _get_score_data(
                Survey.KNOWLEDGE,
                "knowledge",
            ),
            _get_score_data(
                Survey.GENERAL_SELF_EFFICACY,
                "general_self-efficacy",
            ),
            _get_score_data(
                Survey.FEELINGS,
                "specific_self-efficacy",
                columns=get_self_efficacy_feelings_columns(),
            ),
            _get_score_data(
                Survey.APP_RATING,
                "app_rating",
                _get_umars_data,
            ),
            _get_score_data(
                Survey.ACTIONS,
                "hcp_communication",
                lambda time_point: _get_hcp_communication_data(time_point),
            ),
            _get_score_data(
                Survey.ACTIONS,
                "own_medication_change",
                lambda time_point: _get_own_medication_change_data(time_point),
            ),
            _get_score_data(
                Survey.ACTIONS,
                "hcp_medication_change",
                lambda time_point: _get_hcp_medication_change_data(time_point),
            ),
            _get_specific_medication_data(
                "taking_medication",
                "medication_specific_comprehension",
            ),
        ],
    )
    correlation_pairs = [
        ["age", "health_literacy"],
        ["age", "knowledge"],
        ["age", "comprehension"],
        ["age", "app_rating"],
        ["health_literacy", "knowledge"],
        ["health_literacy", "comprehension"],
        ["health_literacy", "general_self-efficacy"],
        ["health_literacy", "specific_self-efficacy"],
        ["health_literacy", "app_rating"],
        ["general_self-efficacy", "knowledge"],
        ["general_self-efficacy", "comprehension"],
        ["general_self-efficacy", "specific_self-efficacy"],
        ["general_self-efficacy", "app_rating"],
        ["general_self-efficacy", "hcp_communication"],
        ["general_self-efficacy", "own_medication_change"],
        ["specific_self-efficacy", "knowledge"],
        ["specific_self-efficacy", "comprehension"],
        ["specific_self-efficacy", "app_rating"],
        ["specific_self-efficacy", "hcp_communication"],
        ["specific_self-efficacy", "own_medication_change"],
        ["knowledge", "comprehension"],
        ["knowledge", "hcp_communication"],
        ["knowledge", "hcp_medication_change"],
        ["knowledge", "own_medication_change"],
        ["comprehension", "hcp_communication"],
        ["comprehension", "hcp_medication_change"],
        ["comprehension", "own_medication_change"],
        *[
            [
                f"taking_medication_{medication}",
                f"medication_specific_comprehension_{medication}",
            ]
            for medication in _get_comprehension_questionnaire_medications()
        ],
    ]
    correlation_result_columns = [
        "First variable",
        "Second variable",
        "Method",
        "Result",
        "Interpretation",
    ]
    means_result_columns = [
        "First variable",
        "Second variable",
        "Correlation",
        "First value",
        "Second mean",
        "Second (n)",
    ]
    correlation_results = []
    means_data = []
    correlations = correlation_data.corr(method="spearman")
    for pair in correlation_pairs:
        first_variable = pair[0]
        second_variable = pair[1]
        first_variable_columns = _get_correlation_columns(
            first_variable,
            correlations,
        )
        second_variable_columns = _get_correlation_columns(
            second_variable,
            correlations,
        )
        for first_variable_column in first_variable_columns:
            for second_variable_column in second_variable_columns:
                if _is_time_point_column(
                    first_variable,
                    first_variable_column,
                ) and _is_time_point_column(
                    second_variable,
                    second_variable_column,
                ):
                    first_time_point = first_variable_column.replace(
                        first_variable,
                        "",
                    )
                    second_time_point = second_variable_column.replace(
                        second_variable,
                        "",
                    )
                    if first_time_point != second_time_point:
                        continue
                correlation_value = correlations.loc[
                    first_variable_column,
                    second_variable_column,
                ]
                correlation_results.append(
                    [
                        _format_correlation_variable(
                            first_variable_column,
                        ),
                        _format_correlation_variable(
                            second_variable_column,
                        ),
                        "⍴",  # noqa: RUF001
                        correlation_value,
                        _get_correlation_interpretation(
                            correlation_value,
                        ).capitalize(),
                    ],
                )
                for first_value in sorted(
                    correlation_data[first_variable_column].dropna().unique(),
                ):
                    value_data = correlation_data[
                        correlation_data[first_variable_column] == first_value
                    ]
                    means_data.append(
                        [
                            _format_correlation_variable(
                                first_variable_column,
                            ),
                            _format_correlation_variable(
                                second_variable_column,
                            ),
                            correlation_value,
                            first_value,
                            value_data[second_variable_column].mean(),
                            len(value_data),
                        ],
                    )

    correlation_table = DataFrame(
        correlation_results,
        columns=correlation_result_columns,
    ).set_index([correlation_result_columns[0], correlation_result_columns[1]])
    means_table = DataFrame(means_data, columns=means_result_columns).set_index(
        [means_result_columns[0], means_result_columns[1]],
    )
    return correlation_table, means_table, correlations, correlation_data
