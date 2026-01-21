"""Formatting utils for analysis outputs.

TODO: refactor dry_run functionality; create data beforehand to be used in
manual time point comparison (e.g., see
compare_baseline_and_follow_up_knowledge) and pass instead of returning
PlotData.
"""

from __future__ import annotations

import colorsys
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, ticker
from pandas import DataFrame, Series
from slugify import slugify

from modules.definitions.constants import (
    FREE_TEXT_DIRECTORY,
    META_COLUMNS,
    MULTIPLE_VALUES_SEPARATOR,
    PARTICIPANT_ID,
    RESULTS_TABLE_FILE,
    SCORE_COLUMN,
)
from modules.definitions.types import (
    Comparison,
    StudyGroup,
    Survey,
    TimePoint,
    format_time_point_name,
)
from modules.survey_results.get_data import (
    UndefinedScoresError,
    filter_results_by_study_group,
    filter_results_by_time_point,
    get_defined_scores,
    get_survey_results,
)
from modules.utils.data import (
    get_label_definition,
    get_score_definition,
    has_multiple_time_points,
    is_free_text_answer,
    is_score_answer,
    load_data_from_file,
    write_data_frame,
)
from modules.utils.output_formatting import (
    STUDY_GROUP_COLORS,
    PlotSizes,
    add_title,
    break_text_after_characters,
    format_float,
    format_output_label,
    show_and_save_plot,
)
from modules.utils.sorting import NOT_ANSWERED_LABEL, sort_by_label
from modules.utils.statistics import (
    ComparisonResult,
    are_study_groups_different_categorical,
    are_study_groups_different_ordinal,
    are_study_groups_different_parametric,
    are_time_points_different_categorical,
    are_time_points_different_ordinal,
    are_time_points_different_parametric,
    get_median_answer,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable


class PlotData:
    """Data used for plotting."""

    data: DataFrame
    column: str
    file_name: str

    def __init__(  # noqa: D107
        self,
        data: DataFrame,
        column: str,
        file_name: str,
        label_definiton: OrderedDict | None,
    ) -> None:
        self.data = data
        self.column = column
        self.file_name = file_name
        self.label_definition = label_definiton


def _get_formatted_mean_score(
    participant_scores: Series,
) -> str:
    """Get formatted mean score."""
    mean_score = participant_scores.mean()
    return f"mean score: {round(mean_score, ndigits=2)}"


def _get_formatted_median_answer(
    participant_data: Series,
    label_definition: OrderedDict,
) -> str:
    median_answer = get_median_answer(
        participant_data,
        label_definition,
    )
    median_answer_string = format_output_label(
        MULTIPLE_VALUES_SEPARATOR.join(median_answer),
        label_definition,
    )
    return break_text_after_characters(
        f"median answer: {median_answer_string}",
        35,
    )


def _get_plot_data(
    data: DataFrame,
    column: str,
    study_group: StudyGroup | None,
    label_definition: OrderedDict | None,
) -> Series:
    relevant_data = (
        filter_results_by_study_group(data, study_group)[column]
        if study_group is not None
        else data[column]
    )
    if label_definition is None:
        return relevant_data.sort_values()
    return relevant_data.fillna(NOT_ANSWERED_LABEL).sort_values(
        key=lambda values: sort_by_label(values, label_definition),
    )


def _format_plot_label(
    label: plt.Text,
    label_definition: OrderedDict | None,
    plot_data: list[Series] | Series,
    show_count: bool = False,  # noqa: FBT001, FBT002
) -> str:
    unformatted_text = label.get_text()
    formatted_text = format_output_label(
        unformatted_text,
        label_definition,
    )
    if show_count:
        if type(plot_data) is not list:
            plot_data = [plot_data]
        counts = [
            f"{(count_data.array == unformatted_text).sum()}"
            for count_data in plot_data
        ]
        label_count = " + ".join(counts)
        formatted_text = f"{formatted_text} ({label_count})"
    return break_text_after_characters(formatted_text, 20, max_breaks=2)


# Based on
# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def _adjust_lightness(
    color_name: str,
    amount: float,
) -> tuple[float, float, float]:
    color = colors.cnames[color_name]
    color = colorsys.rgb_to_hls(*colors.to_rgb(color))
    return colorsys.hls_to_rgb(
        color[0],
        max(0, min(1, amount * color[1])),
        color[2],
    )


def _plot_is_numeric(plot_data: Series) -> bool:
    return pd.api.types.is_any_real_numeric_dtype(plot_data.dtype)


def _get_plot_dimensions(plot_data: Series) -> tuple[float, float, int]:
    if _plot_is_numeric(plot_data):
        return (
            plot_data.min(),
            plot_data.max() + 1,  # not sure why, but + 1 needed
            plot_data.max() - plot_data.min() + 1,
        )
    number_of_values = len(plot_data.unique())
    return 0, number_of_values, number_of_values


# Pretty hacky, should handle with scores if survey was available at this point
# (should refactor)
def _plot_is_ordinal(label_definition: OrderedDict | None) -> bool:
    if label_definition is None:
        return False
    ordinal_indicators = [
        "agree",
        "moderately_true",
        "occasionally",
        "somewhat",
        "neutral",
    ]
    return any(
        ordinal_indicator in label_definition
        for ordinal_indicator in ordinal_indicators
    )


def _maybe_get_study_group_heading_stats(
    plot_data: Series,
    label_definition: OrderedDict,
) -> str | None:
    if _plot_is_numeric(plot_data):
        return _get_formatted_mean_score(plot_data)
    if _plot_is_ordinal(label_definition):
        return _get_formatted_median_answer(
            plot_data,
            label_definition,
        )
    return None


def _maybe_adapt_data_for_stacked_plot(
    plot_data: Series,
    stacked_data: dict[str, DataFrame] | None,
    column: str,
    study_group: StudyGroup,
    label_definition: OrderedDict | None,
) -> tuple[Series | list[Series], str | list[tuple[float, float, float]]]:
    plot_color = (
        STUDY_GROUP_COLORS[study_group]
        if stacked_data is None
        else [
            _adjust_lightness(
                STUDY_GROUP_COLORS[study_group],
                1 + (index / len(stacked_data)),
            )
            for index in range(len(stacked_data))
        ]
    )
    plot_data = (
        plot_data
        if stacked_data is None
        else [
            _get_plot_data(stack_data, column, study_group, label_definition)
            for stack_data in stacked_data.values()
        ]
    )
    return plot_data, plot_color


def _plot_study_group_histogram(  # noqa: PLR0913
    ax: plt.Axes,
    data: DataFrame,
    column: str,
    study_group: StudyGroup,
    label_definition: OrderedDict | None,
    stacked_data: dict[str, DataFrame] | None,
    y_axis_label: str | None,
) -> None:
    """Plot bar in a histogram based on values and study group."""
    plot_data = _get_plot_data(data, column, study_group, label_definition)
    study_group_participants = len(plot_data)
    max_count = plot_data.value_counts().max()
    min_dimension, max_dimension, number_of_bins = _get_plot_dimensions(
        plot_data,
    )
    heading_stats = _maybe_get_study_group_heading_stats(
        plot_data,
        label_definition,
    )
    plot_is_numeric = _plot_is_numeric(plot_data)
    title = f"{study_group.value} (n = {study_group_participants})"
    if heading_stats is not None:
        title = f"{title},\n{heading_stats}"
    plot_data, plot_color = _maybe_adapt_data_for_stacked_plot(
        plot_data,
        stacked_data,
        column,
        study_group,
        label_definition,
    )
    ax.hist(
        plot_data,
        bins=np.arange(min_dimension - PlotSizes.bar_width, max_dimension),
        rwidth=PlotSizes.bar_width,
        color=plot_color,
        stacked=stacked_data is not None,
    )
    if stacked_data is not None:
        ax.legend(labels=stacked_data.keys())
    ax.grid(axis="y", linestyle="-", which="major")
    ax.grid(axis="y", linestyle=":", which="minor")
    if max_count > 40:  # noqa: PLR2004
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    if plot_is_numeric:
        if number_of_bins > 15:  # noqa: PLR2004
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        elif number_of_bins > 12:  # noqa: PLR2004
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.tick_params(axis="x", which="minor", bottom=False)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.xaxis.set_ticks(
        ticks=[label.get_position()[0] for label in ax.xaxis.get_ticklabels()],
        labels=(
            _format_plot_label(label, label_definition, plot_data)
            for label in ax.xaxis.get_ticklabels()
        ),
    )
    ax.set_title(
        title,
        fontsize=PlotSizes.subtitle_font,
    )
    ax.set_ylabel(y_axis_label if y_axis_label is not None else "Participants")
    ax.autoscale()
    if not plot_is_numeric:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")


def _create_group_plots(  # noqa: PLR0913
    title: str,
    data: DataFrame,
    column: str,
    study_groups: list[StudyGroup],
    label_definition: OrderedDict | None,
    file_name: str,
    y_axis_label: str | None = None,
    stacked_data: dict[str, DataFrame] | None = None,
) -> None:
    fig, axs = plt.subplots(1, len(study_groups), sharey="all", layout="tight")
    add_title(title)
    for index, study_group in enumerate(study_groups):
        ax = axs if len(study_groups) == 1 else axs[index]
        _plot_study_group_histogram(
            ax,
            data,
            column,
            study_group,
            label_definition=label_definition,
            y_axis_label=y_axis_label,
            stacked_data=stacked_data,
        )
    show_and_save_plot(fig, file_name)


def _handle_free_text_answer(
    data: DataFrame,
    column: str,
    file_name: str,
) -> None:
    free_text_file_path = f"{FREE_TEXT_DIRECTORY}/{file_name}.txt"
    Path.mkdir(Path(FREE_TEXT_DIRECTORY), exist_ok=True)
    print(  # noqa: T201
        f"ℹ️ Non-aggregated free text answers for '{column}' are saved"  # noqa: RUF001
        f" to {free_text_file_path} (files are not committed)",
    )
    free_text_data = data[column][data[column].notna()]
    separator_line = "\n---\n"
    free_text_lines = [
        "Free text answers",
        separator_line,
    ]
    for line in free_text_data:
        free_text_lines.append(line)
        free_text_lines.append(separator_line)
    with Path.open(free_text_file_path, "w") as free_text_file:
        free_text_file.writelines(free_text_lines)


def _maybe_add_score_description(
    title: str,
    survey: Survey,
    column: str,
) -> str:
    title_parts = [title]
    if is_score_answer(survey, column):
        score_definition = get_score_definition(survey, column)
        if score_definition.description is not None:
            title_parts.append(score_definition.description + ";")
        info_string = score_definition.get_info_string()
        if info_string.lower() not in title.lower():
            title_parts.append(info_string)
    return "\n".join(title_parts)


SHORTEN_LABELS_FOR_SURVEYS = [Survey.APP_RATING]


def _possibly_shorten_labels(
    survey: Survey,
    label_definition: OrderedDict | None,
) -> OrderedDict | None:
    if label_definition is None:
        return label_definition
    if survey in SHORTEN_LABELS_FOR_SURVEYS:
        for key in label_definition:
            label_definition[key] = format_output_label(key, None)
    return label_definition


def _write_to_results_table(
    comparison: Comparison,
    item: str,
    title: str,
    comparison_result: ComparisonResult,
) -> None:
    row_data = [
        comparison.value,
        item,
        title,
        comparison_result.p_value,
        comparison_result.statistic,
        comparison_result.effect_size,
        comparison_result.effect_method,
        comparison_result.notes,
    ]
    if not Path.exists(Path(RESULTS_TABLE_FILE)):
        write_data_frame(
            DataFrame(
                [row_data],
                columns=[
                    "comparison",
                    "item",
                    "title",
                    "p_value",
                    "statistic",
                    "effect_size",
                    "effect_method",
                    "notes",
                ],
            ),
            RESULTS_TABLE_FILE,
        )
        return
    results_data = load_data_from_file(RESULTS_TABLE_FILE)
    present_row = results_data.loc[
        results_data["comparison"].eq(comparison.value)
        & results_data["item"].eq(item)
    ]
    if len(present_row) == 0:
        results_data.loc[len(results_data)] = row_data
    elif len(present_row) == 1:
        results_data.iloc[present_row.index] = row_data
    else:
        print(f"⚠️ Found multiple rows for {comparison.value}, {item}")  # noqa: T201
    write_data_frame(results_data, RESULTS_TABLE_FILE)


def create_comparison_plot(  # noqa: PLR0913
    title: str,
    data: DataFrame,
    column: str,
    label_definition: OrderedDict | None,
    file_name: str,
    dry_run: bool,  # noqa: FBT001
    survey: Survey | None = None,
    y_axis_label: str | None = None,
    stacked_data: dict[str, DataFrame] | None = None,
    assess_difference: bool = True,  # noqa: FBT001, FBT002,
) -> None:
    """Create plot that compares the two study arms."""
    if assess_difference:
        if _plot_is_numeric(data[column]):
            comparison_result = are_study_groups_different_parametric(
                data,
                column,
            )
        elif survey is not None and _plot_is_ordinal(label_definition):
            scores = get_defined_scores(
                survey,
                data[[PARTICIPANT_ID, column]],
            )
            comparison_result = are_study_groups_different_ordinal(
                scores,
                SCORE_COLUMN,
            )
        else:
            if survey is None:
                print(  # noqa: T201
                    "⚠️ Cannot test if survey is ordinal because no survey "
                    "was passed, assuming categorical.",
                )
            comparison_result = are_study_groups_different_categorical(
                data,
                column,
            )
        stats_string = (
            f"p = {format_float(comparison_result.p_value)}, "
            f"{comparison_result.effect_method} = "
            f"{format_float(comparison_result.effect_size)}"
        )
        if not dry_run:
            _write_to_results_table(
                Comparison.STUDY_GROUPS,
                file_name,
                title,
                comparison_result,
            )
        title = f"{title}\n({stats_string})"
    if not dry_run:
        _create_group_plots(
            title,
            data,
            column,
            [StudyGroup.PHARME, StudyGroup.COUNSELING],
            label_definition,
            file_name,
            y_axis_label,
            stacked_data,
        )
    return PlotData(data, column, file_name, label_definition)


def _get_survey_data(
    survey: Survey,
    time_point: TimePoint | None,
) -> DataFrame:
    return (
        get_survey_results(survey)
        if time_point is None
        else filter_results_by_time_point(survey, time_point)
    )


def _get_file_name(
    survey: Survey,
    file_id: str,
    time_point: TimePoint | None,
    subscale_name: str | None,
) -> str:
    file_name_parts = [survey.value.name]
    if time_point is not None:
        file_name_parts.append(time_point.name.lower())
    if subscale_name is not None:
        file_name_parts.append(slugify(subscale_name).lower())
    file_name_parts.append(file_id)
    return "_".join(file_name_parts)


def _get_title(
    main_title: str,
    time_point: TimePoint | None,
    subscale_name: str | None,
) -> str:
    title_parts = []
    if time_point is not None:
        title_parts.append(
            f"[{format_time_point_name(time_point.name).capitalize()}]:",
        )
    title_parts.append(main_title)
    if subscale_name is not None:
        title_parts.append(f"({subscale_name.capitalize()})")
    return " ".join(title_parts)


def _plot_per_question(  # noqa: PLR0913
    survey: Survey,
    columns: list[str] | None,
    time_point: TimePoint | None,
    subscale_name: str | None,
    study_group: StudyGroup | None,
    dry_run: bool,  # noqa: FBT001
) -> list[PlotData] | None:
    data = _get_survey_data(survey, time_point)
    columns = data.columns if columns is None else columns
    column_index = 0
    plot_results = []
    for column in columns:
        if column in META_COLUMNS:
            continue
        column_index += 1
        file_name = _get_file_name(
            survey,
            str(column_index),
            time_point,
            subscale_name,
        )
        title = _maybe_add_score_description(
            _get_title(column, time_point, subscale_name),
            survey,
            column,
        )
        label_definition = _possibly_shorten_labels(
            survey,
            get_label_definition(
                survey,
                column,
            ),
        )
        if is_free_text_answer(survey, column):
            _handle_free_text_answer(data, column, file_name)
        elif study_group is not None and not dry_run:
            _create_group_plots(
                title=title,
                data=data,
                column=column,
                study_groups=[study_group],
                file_name=file_name,
                label_definition=label_definition,
            )
        else:
            plot_results.append(
                create_comparison_plot(
                    title=title,
                    data=data,
                    column=column,
                    file_name=file_name,
                    dry_run=dry_run,
                    label_definition=label_definition,
                    survey=survey,
                ),
            )
    return plot_results if len(plot_results) > 0 else None


def _maybe_map_to_scores(
    data: DataFrame,
    survey: Survey,
    column: str,
) -> DataFrame:
    try:
        score_data = get_defined_scores(
            survey,
            data[[PARTICIPANT_ID, column]],
        )
    except UndefinedScoresError:
        return data
    except Exception:
        raise
    score_data.columns = [PARTICIPANT_ID, column]
    return score_data


def _analyze_time_point_comparison(
    survey: Survey,
    first_data_per_column: list[PlotData],
    second_data_per_column: list[PlotData],
    second_time_point: TimePoint,
    dry_run: bool,  # noqa: FBT001
) -> None:
    for first_data in first_data_per_column:
        column = first_data.column
        label_definition = first_data.label_definition
        second_data = None
        for data in second_data_per_column:
            if data.column == column:
                second_data = data
                break
        if second_data is None:
            message = "Comparison data do not match between time points!"
            raise Exception(message)  # noqa: TRY002
        for study_group in StudyGroup:
            file_name = (
                f"{first_data.file_name}_"
                f"{second_time_point.name.lower()}_{study_group.name.lower()}"
            )
            if _plot_is_numeric(first_data.data[column]):
                comparison_result = are_time_points_different_parametric(
                    first_data.data,
                    second_data.data,
                    column,
                    study_group,
                )
            elif _plot_is_ordinal(label_definition):
                comparison_result = are_time_points_different_ordinal(
                    get_defined_scores(
                        survey,
                        first_data.data[[PARTICIPANT_ID, column]],
                    ),
                    get_defined_scores(
                        survey,
                        second_data.data[[PARTICIPANT_ID, column]],
                    ),
                    SCORE_COLUMN,
                    study_group,
                )
            else:
                comparison_result = are_time_points_different_categorical(
                    _maybe_map_to_scores(first_data.data, survey, column),
                    _maybe_map_to_scores(second_data.data, survey, column),
                    column,
                    study_group,
                )
            if not dry_run:
                _write_to_results_table(
                    Comparison.TIME_POINTS,
                    file_name,
                    first_data.column,
                    comparison_result,
                )


def analyze_time_point_differences(
    survey: Survey,
    plot_data: dict[TimePoint, list[PlotData]],
    dry_run: bool,  # noqa: FBT001
) -> None:
    """Analyze differences between time points."""
    time_point_comparisons = combinations(TimePoint, 2)
    for comparison in time_point_comparisons:
        first_data_per_column = plot_data[comparison[0]]
        second_data_per_column = plot_data[comparison[1]]
        _analyze_time_point_comparison(
            survey,
            first_data_per_column,
            second_data_per_column,
            comparison[1],
            dry_run,
        )


def _maybe_analyze_per_time_point(  # noqa: PLR0913
    create_plots: Callable,
    survey: Survey,
    columns: list[str] | None,
    subscale_name: str | None,
    study_group: StudyGroup | None,
    dry_run: bool,  # noqa: FBT001
) -> any:
    if has_multiple_time_points(survey):
        return_data = {}
        for time_point in TimePoint:
            plot_data = create_plots(
                survey,
                columns,
                time_point,
                subscale_name,
                study_group,
                dry_run,
            )
            return_data[time_point] = plot_data
        analyze_time_point_differences(survey, return_data, dry_run)
        return return_data
    time_point = None
    return create_plots(
        survey,
        columns,
        time_point,
        subscale_name,
        study_group,
        dry_run,
    )


def plot_per_question(
    survey: Survey,
    columns: list[str] | None = None,
    subscale_name: str | None = None,
    study_group: StudyGroup | None = None,
    dry_run: bool = False,  # noqa: FBT001, FBT002
) -> any:
    """Plot survey data per question."""
    return _maybe_analyze_per_time_point(
        _plot_per_question,
        survey,
        columns,
        subscale_name,
        study_group,
        dry_run,
    )


def create_comparison_score_plot(
    title: str,
    score_data: DataFrame,
    file_name: str,
    dry_run: bool,  # noqa: FBT001
    assess_difference: bool = True,  # noqa: FBT001, FBT002
) -> PlotData:
    """Create plot that compares scores."""
    return create_comparison_plot(
        title,
        score_data,
        SCORE_COLUMN,
        file_name=file_name,
        dry_run=dry_run,
        label_definition=None,
        assess_difference=assess_difference,
    )


def _plot_scores(  # noqa: PLR0913
    survey: Survey,
    columns: list[str] | None,
    time_point: TimePoint | None,
    subscale_name: str | None,
    study_group: StudyGroup | None,  # noqa: ARG001, match _plot_per_question
    dry_run: bool,  # noqa: FBT001
) -> list[PlotData]:
    data = _get_survey_data(survey, time_point)
    if columns is not None:
        data = data[[PARTICIPANT_ID, *columns]]
    score_data = get_defined_scores(survey, data)
    main_title_item = (
        survey.value.display_name
        if subscale_name is None
        else subscale_name.capitalize()
    )
    title = _get_title(
        f"{main_title_item} {SCORE_COLUMN}",
        time_point,
        subscale_name=None,
    )
    file_name = _get_file_name(survey, SCORE_COLUMN, time_point, subscale_name)
    return [
        create_comparison_score_plot(
            title,
            score_data,
            file_name,
            dry_run,
        ),
    ]


def plot_scores(
    survey: Survey,
    columns: list[str] | None = None,
    subscale_name: str | None = None,
    dry_run: bool = False,  # noqa: FBT001, FBT002
) -> any:
    """Plot score data per question."""
    return _maybe_analyze_per_time_point(
        _plot_scores,
        survey,
        columns,
        subscale_name,
        study_group=None,
        dry_run=dry_run,
    )


def compare_baseline_and_follow_up_knowledge() -> None:
    """Manually compare baseline knowledge and follow-up knowledge."""
    baseline_score_data = plot_scores(Survey.BASELINE_KNOWLEDGE, dry_run=True)
    baseline_question_data = plot_per_question(
        Survey.BASELINE_KNOWLEDGE,
        dry_run=True,
    )
    follow_up_score_data = plot_scores(Survey.KNOWLEDGE, dry_run=True)
    follow_up_question_data = plot_per_question(
        Survey.KNOWLEDGE,
        dry_run=True,
    )
    for time_point in TimePoint:
        _analyze_time_point_comparison(
            Survey.KNOWLEDGE,
            baseline_score_data,
            follow_up_score_data[time_point],
            time_point,
            dry_run=False,
        )
        _analyze_time_point_comparison(
            Survey.KNOWLEDGE,
            baseline_question_data,
            follow_up_question_data[time_point],
            time_point,
            dry_run=False,
        )
