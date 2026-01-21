"""Code to analyze the completion of testing."""

from pandas import DataFrame

from modules.definitions.constants import (
    CROSSOVER_COMPLETED,
    PARTICIPANT_ID,
    TESTING_COMPLETED,
)
from modules.definitions.types import StudyGroup
from modules.survey_results.get_data import filter_results_by_study_group
from modules.survey_results.redcap_data import get_redcap_data
from modules.utils.output_formatting import format_percentage
from modules.utils.statistics import (
    ALPHA,
    are_study_groups_different_categorical,
)


def _analyze_testing_completion_for_study_group(
    redcap_data: DataFrame,
    study_group: StudyGroup,
) -> str:
    study_group_data = filter_results_by_study_group(
        redcap_data,
        study_group,
    )
    study_group_total = len(study_group_data)
    study_group_not_completed = len(
        study_group_data[study_group_data[TESTING_COMPLETED].isna()],
    )
    study_group_completed = study_group_total - study_group_not_completed
    return (
        f"Completion of testing in {study_group.value} group: "
        f"{study_group_total - study_group_not_completed} "
        f"of {study_group_total} "
        f"({format_percentage(study_group_completed / study_group_total)}%)"
    )


def _analyze_crossover_completion(
    redcap_data: DataFrame,
    study_group: StudyGroup,
) -> str:
    study_group_data = filter_results_by_study_group(
        redcap_data,
        study_group,
    )
    return (
        f"Crossover completed in {study_group.value} group: "
        f"{study_group_data[CROSSOVER_COMPLETED].sum()}"
    )


def analyze_completion_of_testing() -> None:
    """Get completion of testing results per arm."""
    redcap_data = get_redcap_data()
    binary_completion_data = redcap_data[
        [PARTICIPANT_ID, TESTING_COMPLETED]
    ].copy(deep=True)
    binary_completion_data[TESTING_COMPLETED] = binary_completion_data[
        TESTING_COMPLETED
    ].notna()
    comparison_result = are_study_groups_different_categorical(
        binary_completion_data,
        TESTING_COMPLETED,
    )
    print(  # noqa: T201
        _analyze_testing_completion_for_study_group(
            redcap_data,
            StudyGroup.PHARME,
        ),
    )
    print(  # noqa: T201
        _analyze_testing_completion_for_study_group(
            redcap_data,
            StudyGroup.COUNSELING,
        ),
    )
    result = (
        "statistically different"
        if comparison_result.p_value < ALPHA
        else "not statistically different"
    )
    rejected_hypothesis = "H0" if comparison_result.p_value < ALPHA else "H1"
    print(  # noqa: T201
        f"Assuming that the groups are {result}; p = "
        f"{round(comparison_result.p_value, ndigits=3)} "
        f"({rejected_hypothesis} rejected)\n\n",
        _analyze_crossover_completion(redcap_data, StudyGroup.PHARME),
        "(as reported in REDCap)\n",
        _analyze_crossover_completion(redcap_data, StudyGroup.COUNSELING),
    )
