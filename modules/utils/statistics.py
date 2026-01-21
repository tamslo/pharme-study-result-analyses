"""Collection of code to create statistics to keep an overview."""

# Based on https://mverbakel.github.io/2021-02-24/non-inferiority-test
from collections import OrderedDict
from enum import Enum

import numpy as np
import pingouin
from pandas import DataFrame, Series, crosstab
from scipy.stats import (
    MonteCarloMethod,
    chi2_contingency,
    fisher_exact,
    shapiro,
    ttest_ind,
    ttest_ind_from_stats,
)
from statsmodels.stats.contingency_tables import mcnemar

from modules.definitions.constants import PARTICIPANT_ID
from modules.definitions.types import StudyGroup
from modules.survey_results.get_data import filter_results_by_study_group
from modules.utils.sorting import sort_by_label

ALPHA = 0.05
NON_INFERIORITY_ALPHA = ALPHA / 2


class ComparisonResult:
    """Return type for statistical tests."""

    p_value: float
    statistic: str
    effect_size: float
    effect_method: str
    notes: str

    def __init__(  # noqa: D107
        self,
        p_value: float,
        effect_size: float,
        notes: str = "",
    ) -> None:
        self.p_value = p_value
        self.effect_size = effect_size
        self.notes = notes


class TTestResult(ComparisonResult):
    """Return type for a t-test."""

    statistic = "t-test"
    effect_method = "d"


class MannWhitneyUResult(ComparisonResult):
    """Return type for a Mann-Whitney-U test (Wilcoxon rank sum)."""

    statistic = "mannwhitneyu"
    effect_method = "r"


class PairedWilcoxonResult(ComparisonResult):
    """Return type for a Wilcoxon signed rank test."""

    statistic = "wilcoxon"
    effect_method = "r"


class FisherResult(ComparisonResult):
    """Return type for a Fisher's Exact test."""

    statistic = "fisher"
    effect_method = "V"


class McNemarResult(ComparisonResult):
    """Return type for a McNemar test."""

    statistic = "mcnemar"
    effect_method = "ɸ"


def get_median_answer(
    data: Series,
    label_definition: OrderedDict,
) -> list:
    """Get the middle answer or two middle answers from a list of values."""
    sorted_data = data.sort_values(
        key=lambda value: sort_by_label(value, label_definition),
    ).array
    if len(sorted_data) <= 2:  # noqa: PLR2004
        return sorted_data
    if len(sorted_data) % 2 == 1:
        median_index = int((len(sorted_data) - 1) / 2)
        return [sorted_data[median_index]]
    second_index = int(len(sorted_data) / 2)
    first_index = int(second_index - 1)
    if sorted_data[first_index] == sorted_data[second_index]:
        return [sorted_data[first_index]]
    return [
        sorted_data[first_index],
        sorted_data[second_index],
    ]


def _create_comparison_table(
    comparison_data: list[DataFrame],
    column: str,
) -> list[list[int]]:
    comparisons = []
    levels = []
    for data in comparison_data:
        levels += list(data[column].unique())
    for level in set(levels):
        level_counts = []
        for data in comparison_data:
            level_count = data[column].value_counts().get(level, 0)
            level_counts.append(level_count)
        if all(count == 0 for count in level_counts):
            continue
        comparisons.append(level_counts)
    return comparisons


def _get_chi_squared(table: any) -> float:
    return chi2_contingency(table, correction=False)[0]


# From https://www.geeksforgeeks.org/python/how-to-calculate-cramers-v-in-python/
def _get_cramers_v(table: list[list[int]]) -> tuple[float, float]:
    x2 = _get_chi_squared(np.array(table))
    n = np.sum(table)
    minimum_dimension = min(np.array(table).shape) - 1
    if minimum_dimension == 0:
        return x2, float("NaN")
    cramers_v = np.sqrt((x2 / n) / minimum_dimension)
    return x2, cramers_v


# From https://www.askpython.com/python/examples/cohens-d-python
def _get_cohens_d(group1: Series, group2: Series) -> float:
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2),
    )
    return (mean1 - mean2) / pooled_std


def _get_parametric_values(
    data: DataFrame,
    column: str,
    study_group: StudyGroup,
) -> Series:
    group_data = filter_results_by_study_group(data, study_group)
    return group_data[column].dropna().astype(np.int64)


def _is_data_normally_distributed(data: list[Series]) -> bool:
    samples_are_normal = True
    for group_data in data:
        _, p_value = shapiro(group_data)
        group_is_not_normal = p_value < ALPHA
        samples_are_normal = samples_are_normal and not group_is_not_normal
    return samples_are_normal


def are_study_groups_different_parametric(
    data: DataFrame,
    column: str,
) -> TTestResult | MannWhitneyUResult:
    """Test if parametric values of study groups are different."""
    pharme_values = _get_parametric_values(
        data,
        column,
        StudyGroup.PHARME,
    )
    counseling_values = _get_parametric_values(
        data,
        column,
        StudyGroup.COUNSELING,
    )
    samples_are_normal = _is_data_normally_distributed(
        [pharme_values, counseling_values],
    )
    if samples_are_normal:
        print("ℹ️ Using t-test")  # noqa: RUF001, T201
        _, p = ttest_ind(pharme_values, counseling_values)
        return TTestResult(p, _get_cohens_d(pharme_values, counseling_values))
    print("ℹ️ Using Mann-Whitney-U test")  # noqa: RUF001, T201
    return are_study_groups_different_ordinal(data, column)


def are_study_groups_different_ordinal(
    data: DataFrame,
    column: str,
) -> MannWhitneyUResult:
    """Test if ordinal values of study groups are different."""
    pharme_values = _get_parametric_values(
        data,
        column,
        StudyGroup.PHARME,
    )
    counseling_values = _get_parametric_values(
        data,
        column,
        StudyGroup.COUNSELING,
    )
    result = pingouin.mwu(pharme_values, counseling_values)
    return MannWhitneyUResult(
        result.loc["MWU", "p-val"],
        result.loc["MWU", "RBC"],
    )


def are_study_groups_different_categorical(
    data: DataFrame,
    column: str,
) -> FisherResult:
    """Test if categorical values of study groups are different."""
    comparison_data = [
        filter_results_by_study_group(data, study_group)
        for study_group in StudyGroup
    ]
    table = _create_comparison_table(comparison_data, column)
    method = MonteCarloMethod(rng=np.random.default_rng(42))
    result = fisher_exact(table, method=method)
    _, cramers_v = _get_cramers_v(table)
    return FisherResult(result.pvalue, cramers_v)


def _get_paired_data(
    first_data: DataFrame,
    second_data: DataFrame,
    column: str,
) -> tuple[Series, Series]:
    present_first_data = first_data[[PARTICIPANT_ID, column]].dropna()
    present_second_data = second_data[[PARTICIPANT_ID, column]].dropna()
    paired_ids = [
        participant_id
        for participant_id in present_first_data[PARTICIPANT_ID]
        if participant_id in present_second_data[PARTICIPANT_ID].unique()
    ]
    paired_first_data = present_first_data[
        present_first_data[PARTICIPANT_ID].isin(paired_ids)
    ]
    paired_second_data = present_second_data[
        present_second_data[PARTICIPANT_ID].isin(paired_ids)
    ]
    paired_first_data = paired_first_data.set_index(
        PARTICIPANT_ID,
        drop=False,
    ).sort_index()
    paired_second_data = paired_second_data.set_index(
        PARTICIPANT_ID,
        drop=False,
    ).sort_index()
    return paired_first_data, paired_second_data


def are_time_points_different_parametric(
    first_data: DataFrame,
    second_data: DataFrame,
    column: str,
    study_group: StudyGroup,
) -> PairedWilcoxonResult:
    """Test if ordinal data is different between time points.

    Warn if actually parametric (not present in data so far).
    """
    paired_first_data, paired_second_data = _get_paired_data(
        first_data,
        second_data,
        column,
    )
    samples_are_normal = _is_data_normally_distributed(
        [paired_first_data[column], paired_second_data[column]],
    )
    if samples_are_normal:
        message = "‼️ Should use paired t-test"
        raise Exception(message)  # noqa: TRY002
    return are_time_points_different_ordinal(
        first_data,
        second_data,
        column,
        study_group,
    )


def are_time_points_different_ordinal(
    first_data: DataFrame,
    second_data: DataFrame,
    column: str,
    study_group: StudyGroup,
) -> PairedWilcoxonResult:
    """Test if ordinal data is different between time points."""
    paired_first_data, paired_second_data = _get_paired_data(
        first_data,
        second_data,
        column,
    )
    first_data = _get_parametric_values(
        paired_first_data,
        column,
        study_group,
    )
    second_data = _get_parametric_values(
        paired_second_data,
        column,
        study_group,
    )
    result = pingouin.wilcoxon(first_data, second_data)
    return PairedWilcoxonResult(
        result.loc["Wilcoxon", "p-val"],
        result.loc["Wilcoxon", "RBC"],
    )


def are_time_points_different_categorical(
    first_data: DataFrame,
    second_data: DataFrame,
    column: str,
    study_group: StudyGroup,
) -> McNemarResult:
    """Test if categorical data is different between time points."""
    first_study_group_data = filter_results_by_study_group(
        first_data,
        study_group,
    )
    second_study_group_data = filter_results_by_study_group(
        second_data,
        study_group,
    )
    paired_first_data, paired_second_data = _get_paired_data(
        first_study_group_data,
        second_study_group_data,
        column,
    )
    table = crosstab(paired_first_data[column], paired_second_data[column])
    if len(table) == 0:
        return McNemarResult(float("NaN"), float("NaN"), notes="No paired data")
    if len(table) == 1 and len(table.columns) == 1:
        return McNemarResult(1, 1, notes="Same paired data")
    if len(table) > 2 or len(table.columns) > 2:  # noqa: PLR2004
        message = "No 2x2 table; adapt data or implement Bhapkar test!"
        raise Exception(message)  # noqa: TRY002
    significance_result = mcnemar(table)
    # Based on
    # https://koshurai.medium.com/understanding-the-phi-coefficient-a-guide-to-measuring-correlation-between-categorical-variables-741aa9c5acf7
    # (also did manual confirmation)
    chi2 = _get_chi_squared(table)
    phi_coefficient = np.sqrt(chi2 / table.sum().sum())
    return McNemarResult(
        significance_result.pvalue,
        phi_coefficient,
    )


# Based on https://mverbakel.github.io/2021-02-24/non-inferiority-test
# Assuming that an increase in scores is good
def _test_non_inferiority_with_t_test(
    control_group_data: Series,
    test_group_data: Series,
    relative_difference: float,
) -> tuple[float, float]:
    control_group_mean = np.mean(control_group_data)
    delta = relative_difference * control_group_mean
    threshold = control_group_mean - delta
    statistic, two_sided_pvalue = ttest_ind_from_stats(
        mean1=threshold,
        std1=np.std(control_group_data),
        nobs1=len(control_group_data),
        mean2=np.mean(test_group_data),
        std2=np.std(test_group_data),
        nobs2=len(test_group_data),
    )

    pvalue = two_sided_pvalue / 2.0

    return statistic, pvalue


def _test_non_inferiority_with_mann_whitney(
    control_group_data: Series,
    test_group_data: Series,
    relative_difference: float,
) -> float:
    threshold_data = control_group_data.apply(
        lambda value: value - value * relative_difference,
    )
    result = pingouin.mwu(threshold_data, test_group_data, alternative="less")
    return result.loc["MWU", "p-val"]


def test_non_inferiority_between_study_groups(
    data: DataFrame,
    outcome_column: str,
    relative_difference: float,
) -> float:
    """Perform one-sided test with non-inferiority threshold.

    Depending on normality, a t-test or Mann-Whitney-U test are used.

    data: DataFrame with study group and values of interest
    outcome_column: column of interest
    relative_difference: threshold as a percentage of the base group (e.g.,
                         0.1=10% difference)
    """
    control_group_data = filter_results_by_study_group(
        data,
        StudyGroup.COUNSELING,
    )[outcome_column]
    test_group_data = filter_results_by_study_group(data, StudyGroup.PHARME)[
        outcome_column
    ]
    samples_are_normal = _is_data_normally_distributed(
        [control_group_data, test_group_data],
    )
    if samples_are_normal:
        print("ℹ️ Using t-test")  # noqa: RUF001, T201
        return _test_non_inferiority_with_t_test(
            control_group_data,
            test_group_data,
            relative_difference,
        )
    print("ℹ️ Using Mann-Whitney-U test")  # noqa: RUF001, T201
    return _test_non_inferiority_with_mann_whitney(
        control_group_data,
        test_group_data,
        relative_difference,
    )


class EffectInterpretation(Enum):
    """Robust definition of effect interpretations."""

    SMALL = 0
    MEDIUM = 1
    LARGE = 2


EFFECT_MEASURES = ["d", "r"]
ASSOCIATION_MEASURES = ["V", "ɸ"]


def interpret_effect(effect_method: str, effect: float) -> EffectInterpretation:
    """Return effect interpretations."""
    applicable_effect_methods = [*EFFECT_MEASURES, *ASSOCIATION_MEASURES]
    if effect_method not in applicable_effect_methods:
        message = f"Need to add effect interpretations for {effect_method}!"
        raise Exception(message)  # noqa: TRY002
    absolute_effect = abs(effect)
    absolute_upper_interpretation_thresholds = {
        EffectInterpretation.SMALL: 0.3,
        EffectInterpretation.MEDIUM: 0.5,
    }
    if (
        absolute_effect
        > absolute_upper_interpretation_thresholds[EffectInterpretation.MEDIUM]
    ):
        return EffectInterpretation.LARGE
    if (
        absolute_effect
        > absolute_upper_interpretation_thresholds[EffectInterpretation.SMALL]
    ):
        return EffectInterpretation.MEDIUM
    return EffectInterpretation.SMALL
