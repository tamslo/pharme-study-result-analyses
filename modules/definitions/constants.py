"""Constants used throughout the project."""

import json

from dotenv import dotenv_values

from modules.definitions.types import StudyGroup


def get_config() -> dict[str, str]:
    """Read local config from .env file."""
    return dotenv_values(".env")


def get_bool_from_env(key: str) -> bool:
    """Load into JSON because booleans are not parsed."""
    return json.loads(get_config()[key])


SURVEY_DIRECTORY = "data/tasks"
EXTERNAL_DATA_DIRECTORY = "data/external"
SURVEY_DEFINITION_DIRECTORY = "data/dictionary"
DATA_FILE_SUFFIX = ".csv"
PREPROCESSED_FILE_SUFFIX = ".preprocessed"
MANUAL_FILE_SUFFIX = ".manual"
PLOTS_DIRECTORY = "data/plots"
FREE_TEXT_DIRECTORY = "data/free-text"

EHIVE_ID = "ehive_id"
PARTICIPANT_ID = "participant_id"
STUDY_GROUP = "study_group"
TIME_POINT = "authored_at_gmt"
PHARME_ID = "pharme_id"
TESTING_COMPLETED = "testing_completed"
CROSSOVER_COMPLETED = "crossover_completed"
STUDY_ID = "study_id"

SCORE_COLUMN = "score"
PARTICIPANT_SCORE_COLUMNS = [PARTICIPANT_ID, SCORE_COLUMN]

META_COLUMNS = [PARTICIPANT_ID, TIME_POINT, SCORE_COLUMN]

QUESTION_COLUMN = "question"
ANSWER_COLUMN = "answer"
WRONG_ANSWERS_COLUMNS = [PARTICIPANT_ID, QUESTION_COLUMN, ANSWER_COLUMN]
MISSING_GENE_QUESTION = "missing gene"

MULTIPLE_VALUES_SEPARATOR = "|"
SURVEY_TYPE_SEPARATOR = "::"

REDCAP_DATA_FILE = f"{EXTERNAL_DATA_DIRECTORY}/redcap_data{DATA_FILE_SUFFIX}"
PARTICIPANT_ID_MAP_FILE = (
    f"{EXTERNAL_DATA_DIRECTORY}/participant_id_map{DATA_FILE_SUFFIX}"
)

MANUAL_PROGRESS_DATA = (
    f"{EXTERNAL_DATA_DIRECTORY}/participant_surveys.manual.json"
)
PROGRESS_CLEANING_LOG = "data/removed_entries.log"
COMPREHENSION_DATA = f"{EXTERNAL_DATA_DIRECTORY}/comprehension_data.json"
COMPREHENSION_LOG_FILE = "data/wrong_answers.csv"
# Ignored file containing additional information for manual analysis
SECRET_COMPREHENSION_LOG_FILE = "data/wrong_answers_participants.csv"  # noqa: S105
TIME_POINT_PLACEHOLDER = "TIME_POINT"
FILE_NAME_ADDITION_PLACEHOLDER = "FILE_NAME_ADDITION"
COMPREHENSION_SCORE_FILE = (
    f"data/comprehension_scores{FILE_NAME_ADDITION_PLACEHOLDER}"
    f"_{TIME_POINT_PLACEHOLDER}.csv"
)
RESULTS_TABLE_FILE = "data/results_table.csv"

ALL_COMPREHENSION_QUESTIONS_NUMBER = 10
COMPREHENSION_REMOVED_QUESTIONS = [
    "Which of the following genes is not included in your PGx test results?",
]

MULTIPLE_TIME_POINT_SURVEY_NAMES = [
    "actions-taken",
    "comprehension-app",
    "comprehension-counseling",
    "factor-adapted",
    "knowledge-followup",
]

# Because of incomplete preprocessing, the actually normal result is
# shown as indeterminate in PharMe; also applies to PharMe0463, but this
# participant is in the counseling arm and will receive correct data
# once PharMe is activated for them.
DPYD_INDETERMINATE_CASTS = ["PharMe1060"]
# For some participants, PharMe cannot display the whole result as the
# genotype or phenotype are not unambiguous
# This list only includes genes that are included in the comprehension
# questionnaire and PharMe participants
ONLY_GENOTYPE_ADAPTIONS_COMMUNICATED = {
    "PharMe3397": "CYP2C19",
}
ALSO_PHENOTYPE_ADAPTIONS_COMMUNICATED = {}

PREPROCESSING_DIRECTORIES = [
    SURVEY_DIRECTORY,
    EXTERNAL_DATA_DIRECTORY,
]

BASELINE_SURVEY_PROGRESS_FILE = (
    f"{EXTERNAL_DATA_DIRECTORY}/"
    f"particitpants_surveys_baseline_pharme{PREPROCESSED_FILE_SUFFIX}.csv"
)
PHARME_GROUP_SURVEY_PROGRESS_FILE = (
    f"{EXTERNAL_DATA_DIRECTORY}/"
    f"particitpants_surveys_case_pharme{PREPROCESSED_FILE_SUFFIX}.csv"
)
COUNSELING_GROUP_SURVEY_PROGRESS_FILE = (
    f"{EXTERNAL_DATA_DIRECTORY}/"
    f"particitpants_surveys_control_pharme{PREPROCESSED_FILE_SUFFIX}.csv"
)
PROGRESS_DATA_FILE = f"{EXTERNAL_DATA_DIRECTORY}/progress_data.csv"

MANUAL_FILE_PATHS = [
    COMPREHENSION_DATA,
    BASELINE_SURVEY_PROGRESS_FILE.replace(PREPROCESSED_FILE_SUFFIX, ""),
    PHARME_GROUP_SURVEY_PROGRESS_FILE.replace(PREPROCESSED_FILE_SUFFIX, ""),
    COUNSELING_GROUP_SURVEY_PROGRESS_FILE.replace(PREPROCESSED_FILE_SUFFIX, ""),
    SURVEY_DIRECTORY,
]

ROBUST_GENERATED_DATA_FILES = [
    REDCAP_DATA_FILE,
    PROGRESS_DATA_FILE,
    PROGRESS_CLEANING_LOG,
    COMPREHENSION_LOG_FILE,
    SECRET_COMPREHENSION_LOG_FILE,
]

GENERATED_DATA_FILES = [
    PARTICIPANT_ID_MAP_FILE,
    *ROBUST_GENERATED_DATA_FILES,
]

REDCAP_STUDY_GROUPS = {
    "0": StudyGroup.COUNSELING,
    "1": StudyGroup.PHARME,
}
