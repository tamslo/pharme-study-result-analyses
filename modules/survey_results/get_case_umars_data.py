"""Code to get case uMARS surveys from REDCap.

Since no manual uMARS surveys exist, we are writing the REDCap data to the
manual uMARS file as a workaround.
"""

import json
from pathlib import Path

import pandas as pd

from modules.definitions.constants import MANUAL_PROGRESS_DATA, META_COLUMNS
from modules.definitions.types import StudyGroup, Survey, TimePoint
from modules.survey_results.get_data import get_manual_progress_data
from modules.survey_results.redcap_data import get_study_group_for_redcap_user
from modules.utils.anonymization import get_participant_id_map
from modules.utils.data import (
    get_data_path,
    get_definition_data_path,
    get_manual_path,
    load_answer_definitions,
    load_data_from_file,
    write_data_frame,
)
from modules.utils.redcap import get_redcap_users


def _get_initial_umars_path() -> Path:
    return get_data_path(Survey.APP_RATING, preprocessed=False)


def _get_manual_umars_path() -> Path:
    return get_manual_path(_get_initial_umars_path())


def _initialize_case_umars_data() -> pd.DataFrame:
    file_path = _get_manual_umars_path()
    if file_path.exists():
        return load_data_from_file(file_path)
    present_umars_data = load_data_from_file(_get_initial_umars_path())
    return pd.DataFrame(columns=present_umars_data.columns)


REDCAP_UMARS_MAPPING = {
    "Entertainment": "sec_a_1",
    "Interest": "sec_a_2",
    "Customisation": "sec_a_3",
    "Interactivity": "sec_a_4",
    "Target group": "sec_a_5",
    "Performance": "sec_6",
    "Ease of use": "sec_7",
    "Navigation": "sec_8",
    "Gestural design": "sec_9",
    "Layout": "sec_10",
    "Graphics": "sec_11",
    "Quality of information": "sec_13",
    "Quantity of information": "sec_14",
    "Visual information": "sec_15",
    "Credibility of source": "sec_16",
    (
        "Would you recommend this app to people who might benefit from it?"
    ): "sec_17",
    (
        "How many times do you think you would use this app in the next 12 "
        "months if it was relevant to you?"
    ): "sec_18",
    "Would you pay for this app?": "sec_19",
    "What is your overall (star) rating of the app?": "sec_20",
    "Awareness": "sec_21",
    "Knowledge": "sec_22",
    "Attitudes": "sec_23",
    "Intention to change": "sec_24",
    "Help seeking": "sec_25",
    "Behaviour change": "sec_26",
    "Visual appeal": "sec_12",
    "Further comments about the app?": "sec_27",
}


def maybe_update_case_umars_data() -> None:  # noqa: C901, PLR0912
    """Update case uMARS data, if applicable."""
    survey = Survey.APP_RATING
    case_umars_data = _initialize_case_umars_data()
    redcap_users = get_redcap_users()
    column_definitions = load_data_from_file(
        get_definition_data_path(survey),
    )
    manual_progress_data = get_manual_progress_data()
    participant_id_map = get_participant_id_map()
    for redcap_user in redcap_users:
        if redcap_user["app_rating_survey_complete"] == "1":
            print(  # noqa: T201
                "⚠️ Partial uMARS surveys are not yet implemented since "
                "it was not needed.",
            )
            continue
        if redcap_user["app_rating_survey_complete"] == "2":
            ehive_id = redcap_user["ehive_id"]
            if ehive_id in case_umars_data["participant_id"]:
                continue
            if (
                get_study_group_for_redcap_user(redcap_user)
                != StudyGroup.COUNSELING
            ):
                message = (
                    "‼️ uMARS survey for REDCap user "
                    f"{redcap_user['study_id']} and not in counseling group "
                    "is skipped"
                )
                print(message)  # noqa: T201
                continue
            if redcap_user["pharme_data_uploaded"] == "":
                message = (
                    "‼️ uMARS survey for REDCap user "
                    f"{redcap_user['study_id']} who did not set up PharMe "
                    "is skipped"
                )
                print(message)  # noqa: T201
                continue
            # Using EOS date as survey date as a workaround
            survey_date = redcap_user["eos_date"]
            participant_data = [ehive_id, survey_date, None]
            data_columns = [
                column
                for column in case_umars_data.columns
                if column not in META_COLUMNS
            ]
            for column in data_columns:
                column_definition = column_definitions[
                    column_definitions["title"] == column
                ]
                column_type = column_definition["type"].array[0]
                response = redcap_user[REDCAP_UMARS_MAPPING[column]]
                if column_type == "SINGLE_CHOICE":
                    answer_definitions = load_answer_definitions(
                        survey,
                        column,
                    )
                    score = int(response)
                    # N/A included
                    if len(answer_definitions) == 6:  # noqa: PLR2004
                        score -= 1
                    answer_definition = next(
                        (
                            answer_definition
                            for answer_definition in answer_definitions
                            if answer_definition["score"] == score
                        ),
                        None,
                    )
                    if answer_definition is None:
                        participant_data.append(None)
                    else:
                        participant_data.append(answer_definition["key"])
                else:
                    participant_data.append(response)
            case_umars_data.loc[len(case_umars_data)] = participant_data
            anonymous_id = participant_id_map[ehive_id]
            time_point_name = (
                f"{survey.value.name}_{TimePoint.RESULT_RETURN.value.postfix}"
            )
            if anonymous_id in manual_progress_data:
                manual_progress_data[anonymous_id][time_point_name] = (
                    survey_date
                )
            else:
                manual_progress_data[anonymous_id] = {
                    time_point_name: survey_date,
                }
    write_data_frame(case_umars_data, _get_manual_umars_path())
    with Path.open(MANUAL_PROGRESS_DATA, "w") as manual_progress_file:
        json.dump(manual_progress_data, manual_progress_file, indent=4)
