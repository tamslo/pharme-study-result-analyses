"""Data preprocessing for satisfaction data."""

from modules.definitions.types import Survey
from modules.survey_results.normalize import (
    get_normalized_survey_data,
)
from modules.utils.data import get_data_path, write_data_frame

REPLACE_SATISFACTION_APP_COLUMN_FORMULATION = " with the PharMe app"
REPLACE_SATISFACTION_COLUMN_FORMULATIONS = [
    " with the pharmacist",
    REPLACE_SATISFACTION_APP_COLUMN_FORMULATION,
]
SATISFACTION_COLUMN_REPLACEMENT = " with the pharmacist or the PharMe app"


def normalize_satisfaction_surveys() -> None:
    """Normalize satisfaction surveys."""
    write_data_frame(
        get_normalized_survey_data(
            Survey.SATISFACTION_COUNSELING,
            Survey.SATISFACTION_APP,
            Survey.SATISFACTION,
            REPLACE_SATISFACTION_COLUMN_FORMULATIONS,
            SATISFACTION_COLUMN_REPLACEMENT,
        ),
        get_data_path(Survey.SATISFACTION),
    )
