"""Formatting utils for analysis outputs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from modules.definitions.constants import (
    MULTIPLE_VALUES_SEPARATOR,
    PLOTS_DIRECTORY,
    RESULTS_TABLE_FILE,
)
from modules.definitions.types import (
    Comparison,
    StudyGroup,
)
from modules.utils.data import load_data_from_file
from modules.utils.statistics import (
    ALPHA,
    interpret_effect,
)

if TYPE_CHECKING:
    from collections import OrderedDict

    from pandas import DataFrame


STUDY_GROUP_COLORS = {
    StudyGroup.PHARME: "purple",
    StudyGroup.COUNSELING: "orange",
}


class PlotSizes:
    """Constants for text sizes in plots."""

    title_font: int = 14
    subtitle_font: int = 12
    bar_width: float = 0.5


def format_float(value: float, ndigits: int = 2) -> float:
    """Round float to number of digits (default 2)."""
    rounded_value = round(value, ndigits=ndigits)
    while rounded_value == 0.0:
        ndigits += 1
        rounded_value = round(value, ndigits=ndigits)
    return rounded_value


def format_percentage(number: float) -> int | float:
    """Convert to percent and round to maximal one digit."""
    percent = number * 100
    if percent.is_integer():
        return int(percent)
    return round(percent, ndigits=1)


def _format_single_label(
    text: str,
    label_definition: OrderedDict | None,
) -> str:
    """Format a label without MULTIPLE_VALUES_SEPARATOR."""
    if label_definition is None or text not in label_definition:
        return text.replace("_", " ").capitalize()
    return label_definition[text]


def format_output_label(
    text: str,
    label_definition: OrderedDict | None,
) -> str:
    """Format the text of an output value."""
    if MULTIPLE_VALUES_SEPARATOR in text:
        return ", ".join(
            [
                _format_single_label(
                    single_label,
                    label_definition,
                )
                for single_label in text.split(MULTIPLE_VALUES_SEPARATOR)
            ],
        )
    return _format_single_label(text, label_definition)


def break_text_after_characters(
    text: str,
    preferred_characters: int,
    max_breaks: float = math.inf,
) -> str:
    """Break text at spaces after characters."""
    required_breaks = len(text) // preferred_characters
    if "\n" in text or required_breaks == 0:
        return text
    while required_breaks > max_breaks:
        preferred_characters += 1
        required_breaks = len(text) // preferred_characters
    text = text.replace("/", "/ ")
    space_positions = list(re.finditer(" ", text))
    breaking_spaces = []
    for break_number in range(1, required_breaks + 1):
        space_index = round(
            len(space_positions) * (break_number / (required_breaks + 1)),
        )
        breaking_space = space_positions[space_index]
        breaking_spaces.append(breaking_space)
    current_space = breaking_spaces[0]
    next_space_index = 1
    multiline_text = [text[0 : current_space.start()]]
    for next_space_index in range(1, len(breaking_spaces) + 1):
        if next_space_index == len(breaking_spaces):
            multiline_text.append(text[current_space.end() :])
        else:
            next_space = breaking_spaces[next_space_index]
            multiline_text.append(
                text[current_space.end() : next_space.start()],
            )
            current_space = next_space
    multiline_text = "\n".join(multiline_text)
    return multiline_text.replace("/ ", "/")


def show_and_save_plot(fig: plt.Figure, file_name: str) -> None:
    """Show created plot and save the figure to PDF."""
    plt.show()
    Path(PLOTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{PLOTS_DIRECTORY}/{file_name}.pdf", bbox_inches="tight")


def add_title(title: str) -> None:
    """Add a plot tile."""
    plt.suptitle(
        title,
        fontsize=PlotSizes.title_font,
        weight="bold",
        wrap=True,
    )


def get_interesting_comparison_results(comparison: Comparison) -> DataFrame:
    """Get filtered significant results or those with high effect."""
    results_data = load_data_from_file(RESULTS_TABLE_FILE)
    comparison_data = results_data[
        results_data["comparison"] == comparison.value
    ].copy(deep=True)
    significant_data = comparison_data["p_value"] < ALPHA
    comparison_data["is_significant"] = significant_data
    significant_indices = comparison_data.index[significant_data]
    effect_interpretations = [
        interpret_effect(
            iterator[1]["effect_method"],
            iterator[1]["effect_size"],
        )
        for iterator in comparison_data.iterrows()
    ]
    comparison_data["effect_interpretation"] = [
        effect_interpretation.name.lower()
        for effect_interpretation in effect_interpretations
    ]
    return comparison_data.loc[significant_indices]
