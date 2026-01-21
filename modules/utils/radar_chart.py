"""Code to generate a radar plot.

Based on the official documentation, see
https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html.

"""

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from modules.definitions.constants import PARTICIPANT_ID
from modules.definitions.types import StudyGroup, Survey
from modules.survey_results.get_data import get_single_score, get_survey_results
from modules.survey_results.redcap_data import get_study_group
from modules.utils.data import is_score_answer, value_is_nan
from modules.utils.output_formatting import (
    STUDY_GROUP_COLORS,
    add_title,
    format_float,
    show_and_save_plot,
)


def _radar_factory(  # noqa: ANN202, C901
    num_vars: int,
    frame: str = "circle",
):
    """Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):  # noqa: ANN001, ANN202
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:  # noqa: SLF001
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)  # noqa: B026

        def plot(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line) -> None:  # noqa: ANN001
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up  # noqa: FIX001, TD001
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels) -> None:  # noqa: ANN001
            _, labels = self.set_thetagrids(np.degrees(theta), labels)
            for label in labels:
                x_position = label.get_position()[0]
                if x_position > 0 and x_position < math.pi:
                    label.set_horizontalalignment("right")
                if x_position > math.pi:
                    label.set_horizontalalignment("left")

        # From https://stackoverflow.com/a/52911181
        def draw(self, renderer) -> None:  # noqa: ANN001
            """Draw. If frame is polygon, make gridlines polygon-shaped."""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars  # noqa: SLF001
            super().draw(renderer)

        def _gen_axes_patch(self) -> None:
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            if frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5),
                    num_vars,
                    radius=0.5,
                    edgecolor="k",
                )
            message = f"Unknown value for 'frame': {frame}"
            raise ValueError(message)

        def _gen_axes_spines(self) -> None:
            if frame == "circle":
                return super()._gen_axes_spines()
            if frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes,
                )
                return {"polar": spine}
            message = f"Unknown value for 'frame': {frame}"
            raise ValueError(message)

    register_projection(RadarAxes)
    return theta


def _create_radar_chart(  # noqa: PLR0913
    title: str,
    data: list[list[float]],
    mean_values: list[float],
    dimensions: list[str],
    study_groups: list[StudyGroup],
    file_name: str,
) -> None:
    """Create radar chart (or spider chart)."""
    theta = _radar_factory(len(dimensions), frame="polygon")
    fig, ax = plt.subplots(subplot_kw={"projection": "radar"}, layout="tight")
    add_title(title)
    alpha = 0.03
    for values_index, values in enumerate(data):
        color = STUDY_GROUP_COLORS[study_groups[values_index]]
        ax.plot(
            theta,
            values,
            color=color,
            alpha=0,
        )
        ax.fill(
            theta,
            values,
            facecolor=color,
            alpha=alpha,
        )
    dimensions_with_mean = []
    for dimension_index, dimension in enumerate(dimensions):
        mean_value = mean_values[dimension_index]
        dimensions_with_mean.append(
            f"{dimension}\n(mean: {format_float(mean_value)})",
        )
    ax.plot(theta, mean_values, color="white")
    ax.set_varlabels(dimensions_with_mean)
    data_values = pd.Series(
        [
            value if not value_is_nan(value) else 0
            for value_list in data
            for value in value_list
        ],
    ).unique()
    grid_steps = range(
        math.floor(data_values.min()),
        math.ceil(data_values.max()) + 1,
        1,
    )
    _, y_labels = ax.set_rgrids(grid_steps)
    magic_alignment_number = 0.4
    x_position = theta[len(theta) - 1] - magic_alignment_number
    for label in y_labels:
        label_value = float(label.get_text())
        if label_value == 0.0:
            label.set_text("N/A")
        if label_value < mean_values[len(mean_values) - 1]:
            label.set_color("white")
        y_position = label.get_position()[1]
        label.set_position(
            (x_position, y_position),
        )
    ax.set_axisbelow(False)
    ax.set_yticklabels(y_labels)
    show_and_save_plot(
        fig,
        file_name,
    )


def get_radar_chart_mean(data: list[float], zeros_are_na: bool) -> float:  # noqa: FBT001
    """Get mean value for radar data."""
    data_series = pd.Series(data)
    if zeros_are_na:
        return data_series[data_series > 0].mean()
    return data_series.mean()


def get_radar_chart_means(
    data: list[list[float]],
    columns: list[str],
    zeros_are_na: bool,  # noqa: FBT001
) -> list[float]:
    """Get mean values per column.

    N/A (0) can be ignored for mean scores.
    """
    mean_values = []
    for column_index, _ in enumerate(columns):
        column_values = [single_list[column_index] for single_list in data]
        mean_value = get_radar_chart_mean(column_values, zeros_are_na)
        mean_values.append(mean_value)
    return mean_values


def get_radar_chart_data(
    survey: Survey,
    columns: list[str],
) -> tuple[list[list[float]], list[StudyGroup], list[str]]:
    """Get data formatted for a radar chart.

    N/A is counted as 0.
    """
    survey_data = get_survey_results(survey)
    data = []
    study_groups = []
    participant_ids = []
    for _, row in survey_data.iterrows():
        participant_data = []
        for column in columns:
            answer = row[column]
            if is_score_answer(survey, column):
                participant_data.append(answer)
            else:
                score = get_single_score(survey, column, answer)
                if score is None:
                    score = 0
                participant_data.append(score)
        data.append(participant_data)
        study_groups.append(get_study_group(row[PARTICIPANT_ID]))
        participant_ids.append(row[PARTICIPANT_ID])
    return data, study_groups, participant_ids


def create_radar_chart(  # noqa: PLR0913
    survey: Survey,
    data: list[list[float]],
    subscale: str,
    columns: list[str],
    study_groups: list[StudyGroup],
    zeros_are_na: bool,  # noqa: FBT001
    title_addition: str = "",
) -> None:
    """Create radar chart (or spider chart) based on a survey."""
    mean_values = get_radar_chart_means(data, columns, zeros_are_na)
    title = (
        f"{survey.value.display_name}: {subscale.capitalize()}{title_addition}"
    )
    _create_radar_chart(
        title=title,
        data=data,
        mean_values=mean_values,
        dimensions=columns,
        study_groups=study_groups,
        file_name=f"{survey.value.name}_{subscale.lower()}_spider_chart",
    )
