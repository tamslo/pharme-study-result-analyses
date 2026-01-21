"""Code to analyze self-efficacy."""

from pandas import Index

from modules.definitions.types import Survey
from modules.survey_results.get_data import (
    get_survey_results,
)


def get_self_efficacy_feelings_columns() -> list[Index]:
    """Get self efficacy columns from feelings data."""
    feelings_data = get_survey_results(Survey.FEELINGS)
    feelings_columns = feelings_data.columns
    return feelings_columns[3:6]
