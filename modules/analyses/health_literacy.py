"""Code to analyze baseline criteria."""

from collections import OrderedDict

from pandas import DataFrame

from modules.definitions.constants import PARTICIPANT_ID, SCORE_COLUMN
from modules.definitions.types import Survey
from modules.survey_results.get_data import get_survey_results
from modules.utils.data import get_score_interpretation

HEALTH_LITERACY_TITLE = "Health literacy"
HEALTH_LITERACY_COLUMN = "health_literacy"
# Scoring based on Haun et al., 2012
# "(a) inadequate (4-12); (b) marginal (13-16); and adequate (17-20)""
BRIEF_SCORE_INTERPRETATION = OrderedDict(
    [
        ("inadequate", 12),
        ("marginal", 16),
        ("adequate", 20),
    ],
)
HEALTH_LITERACY_LABELS = OrderedDict(
    [
        ("inadequate", "Inadequate"),
        ("marginal", "Marginal"),
        ("adequate", "Adequate"),
    ],
)


def get_health_literacy_scores() -> DataFrame:
    """Get health literacy scores."""
    health_literacy_data = get_survey_results(
        Survey.HEALTH_LITERACY,
    )
    interpreted_scores = []
    for _, row in health_literacy_data.iterrows():
        participant_id = row[PARTICIPANT_ID]
        brief_score = row[SCORE_COLUMN]
        interpreted_scores.append(
            [
                participant_id,
                get_score_interpretation(
                    brief_score,
                    BRIEF_SCORE_INTERPRETATION,
                ),
            ],
        )
    return DataFrame(
        interpreted_scores,
        columns=[PARTICIPANT_ID, HEALTH_LITERACY_COLUMN],
    )
