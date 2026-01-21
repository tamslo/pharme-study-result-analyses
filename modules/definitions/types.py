"""Types used throughout the project."""

from __future__ import annotations

from enum import Enum


class ThisShouldNeverHappenError(Exception):
    """Use this when checking for things that are not expected to happen."""


class SurveyType(Enum):
    """Defines which dictionary to choose."""

    LIBRARY = "library"
    PHARME = "pharme"


class SurveyDefinition:
    """Collects all important properties for accessing surveys."""

    name: str
    type: SurveyType
    is_final: bool
    display_name: str

    def __init__(  # noqa: D107
        self,
        name: str,
        survey_type: SurveyType,
        is_final: bool = True,  # noqa: FBT001, FBT002
        display_name: str = "",
    ) -> None:
        self.name = name
        self.type = survey_type
        self.is_final = is_final
        self.display_name = (
            name.replace("-", " ").capitalize()
            if display_name == ""
            else display_name
        )


class Survey(Enum):
    """Robust definition of surveys and their filenames."""

    DEMOGRAPHICS = SurveyDefinition("demographics", SurveyType.PHARME)
    COMPREHENSION_APP = SurveyDefinition(
        "comprehension-app",
        SurveyType.PHARME,
        is_final=False,
    )
    COMPREHENSION_COUNSELING = SurveyDefinition(
        "comprehension-counseling",
        SurveyType.PHARME,
        is_final=False,
    )
    COMPREHENSION = SurveyDefinition("comprehension", SurveyType.PHARME)
    HEALTH_LITERACY = SurveyDefinition("brief", SurveyType.LIBRARY)
    GENERAL_SELF_EFFICACY = SurveyDefinition(
        "gse",
        SurveyType.PHARME,
        display_name="GSE",
    )
    BASELINE_KNOWLEDGE = SurveyDefinition(
        "knowledge",
        SurveyType.PHARME,
        display_name="Knowledge (baseline)",
    )
    KNOWLEDGE = SurveyDefinition(
        "knowledge-followup",
        SurveyType.PHARME,
        display_name="Knowledge",
    )
    SATISFACTION_APP = SurveyDefinition(
        "satisfaction-app",
        SurveyType.PHARME,
        is_final=False,
    )
    SATISFACTION_COUNSELING = SurveyDefinition(
        "satisfaction-counseling",
        SurveyType.PHARME,
        is_final=False,
    )
    SATISFACTION = SurveyDefinition("satisfaction", SurveyType.PHARME)
    ACTIONS = SurveyDefinition("actions-taken", SurveyType.PHARME)
    ATTITUDES = SurveyDefinition("attitudes", SurveyType.PHARME)
    FEELINGS = SurveyDefinition("factor-adapted", SurveyType.PHARME)
    APP_RATING = SurveyDefinition(
        "u-mars",
        SurveyType.PHARME,
        display_name="App rating",
    )


class StudyGroup(Enum):
    """Study group constants."""

    PHARME = "PharMe"
    COUNSELING = "Counseling"


class TimePointDefinition:
    """Collects all important properties for a time point."""

    postfix: str | None
    index: int

    def __init__(self, postfix: str, index: int) -> None:  # noqa: D107
        self.postfix = postfix
        self.index = index


class TimePoint(Enum):
    """Definition of study time points and indices."""

    RESULT_RETURN = TimePointDefinition("t0", 0)
    ONE_MONTH_FOLLOW_UP = TimePointDefinition("t30", 1)
    THREE_MONTH_FOLLOW_UP = TimePointDefinition("t90", 2)


class Comparison(Enum):
    """Definition of available comparisons."""

    STUDY_GROUPS = "study_groups"
    TIME_POINTS = "time_points"


BASELINE_SURVEYS = [
    Survey.DEMOGRAPHICS,
    Survey.HEALTH_LITERACY,
    Survey.GENERAL_SELF_EFFICACY,
    Survey.BASELINE_KNOWLEDGE,
]


def format_time_point_name(time_point_name: str) -> str:
    """Format the name of a TimePoint (e.g., for logging)."""
    return (
        time_point_name.lower()
        .replace("_", " ")
        .replace(" month", "-month")
        .replace("follow up", "follow-up")
    )
