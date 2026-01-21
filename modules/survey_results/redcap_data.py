"""Definition of study groups and code to update data from REDcap."""

from __future__ import annotations

import logging
from pathlib import Path

from pandas import DataFrame

from modules.definitions.constants import (
    CROSSOVER_COMPLETED,
    PARTICIPANT_ID,
    REDCAP_DATA_FILE,
    REDCAP_STUDY_GROUPS,
    STUDY_GROUP,
    TESTING_COMPLETED,
    get_bool_from_env,
)
from modules.definitions.types import StudyGroup
from modules.utils.anonymization import get_participant_id_map, reveal_ehive_id
from modules.utils.data import (
    load_data_from_file,
    value_is_nan,
    write_data_frame,
)
from modules.utils.redcap import get_redcap_users


def _get_redcap_user_with_ehive_id(
    redcap_users: list[dict],
    ehive_id: str,
) -> dict | None:
    return next(
        (user for user in redcap_users if user["ehive_id"] == ehive_id),
        None,
    )


def get_study_group_for_redcap_user(redcap_user: dict) -> StudyGroup | None:
    """Get the study group for a REDCap user."""
    redcap_study_group = redcap_user["randomization"]
    if redcap_study_group == "":
        return None
    return REDCAP_STUDY_GROUPS[redcap_study_group]


def _get_study_group(
    redcap_users: list[dict],
    ehive_id: str,
) -> StudyGroup | None:
    redcap_user = _get_redcap_user_with_ehive_id(redcap_users, ehive_id)
    if redcap_user is None:
        return None
    return get_study_group_for_redcap_user(redcap_user)


def _get_testing_completed(
    redcap_users: list[dict],
    ehive_id: str,
    study_group: StudyGroup | None,
) -> True | None:
    if study_group is None:
        return None
    redcap_user = _get_redcap_user_with_ehive_id(redcap_users, ehive_id)
    if redcap_user is None:
        return None
    if study_group is StudyGroup.PHARME:
        testing_completed = redcap_user["pharme_data_uploaded"]
    else:
        testing_completed = redcap_user["counsel_date"]
    if testing_completed == "":
        return None
    return True


def _get_crossover_completed(
    redcap_users: list[dict],
    ehive_id: str,
    study_group: StudyGroup | None,
) -> bool | None:
    if study_group is None:
        return None
    redcap_user = _get_redcap_user_with_ehive_id(redcap_users, ehive_id)
    if redcap_user is None:
        return None
    if study_group is StudyGroup.PHARME:
        return redcap_user["crossover_complete"] == "2"
    other_study_group = StudyGroup.PHARME
    other_intervention_completed = _get_testing_completed(
        redcap_users,
        ehive_id,
        other_study_group,
    )
    return other_intervention_completed is not None


def _initialize_redcap_data() -> DataFrame:
    return DataFrame(
        columns=[
            PARTICIPANT_ID,
            STUDY_GROUP,
            TESTING_COMPLETED,
            CROSSOVER_COMPLETED,
        ],
    )


def _get_study_group_string(
    redcap_data: DataFrame,
    participant_id: str,
) -> str:
    return redcap_data[redcap_data[PARTICIPANT_ID] == participant_id][
        STUDY_GROUP
    ].to_list()[0]


def get_study_group(participant_id: str) -> StudyGroup:
    """Get study group for a single participant."""
    redcap_data = get_redcap_data()
    study_group_string = _get_study_group_string(redcap_data, participant_id)
    if value_is_nan(study_group_string):
        return None
    return next(
        study_group
        for study_group in StudyGroup
        if study_group.value == study_group_string
    )


def get_redcap_data() -> DataFrame:
    """Get study groups if they exist."""
    redcap_data_file = Path(REDCAP_DATA_FILE)
    if not redcap_data_file.exists():
        return _initialize_redcap_data()
    return load_data_from_file(REDCAP_DATA_FILE)


def redcap_data_are_complete() -> bool:
    """Check if all participants have a study group assigned."""
    redcap_data = get_redcap_data()
    redcap_data_are_incomplete = (
        redcap_data is None
        or redcap_data.empty
        or any(redcap_data[STUDY_GROUP].isna())
        or any(redcap_data[TESTING_COMPLETED].isna())
    )
    return not redcap_data_are_incomplete


def maybe_update_redcap_data(
    participant_ids: list[str],
) -> None:
    """Update data from REDCap."""
    redcap_data = get_redcap_data()
    redcap_data_are_valid = all(
        participant_id in participant_ids
        for participant_id in redcap_data[PARTICIPANT_ID].array
    )
    if not redcap_data_are_valid:
        redcap_data = _initialize_redcap_data()
    all_participants_are_covered = all(
        participant_id in redcap_data[PARTICIPANT_ID].array
        for participant_id in participant_ids
    )
    logger = logging.getLogger(__name__)
    if all_participants_are_covered:
        data_are_complete = redcap_data_are_complete()
        if data_are_complete:
            logger.info(
                "Loaded REDcap data from file, the data are complete for "
                "present participants",
            )
            return
        update_if_incomplete = get_bool_from_env(
            "UPDATE_DATA_FROM_REDCAP_IF_INCOMPLETE",
        )
        if not update_if_incomplete:
            logger.info("Loaded REDcap data from file")
            logger.warning(
                "⚠️  Not updating REDcap data despite potentially "
                "incomplete records due to .env setting",
            )
            return
    logger.info("Updating REDcap data...")
    redcap_users = get_redcap_users()
    participant_id_map = get_participant_id_map()
    next_index = len(redcap_data.index)
    for participant_id in participant_ids:
        present_ids = redcap_data[PARTICIPANT_ID].array
        if participant_id in present_ids:
            data_index = redcap_data[
                redcap_data[PARTICIPANT_ID] == participant_id
            ].index
        else:
            data_index = next_index
            next_index += 1
        ehive_id = reveal_ehive_id(participant_id_map, participant_id)
        study_group = _get_study_group(
            redcap_users,
            ehive_id,
        )
        testing_completed = _get_testing_completed(
            redcap_users,
            ehive_id,
            study_group,
        )
        crossover_completed = _get_crossover_completed(
            redcap_users,
            ehive_id,
            study_group,
        )
        redcap_data.loc[data_index] = [
            participant_id,
            study_group.value if study_group is not None else None,
            testing_completed,
            crossover_completed,
        ]
    write_data_frame(redcap_data, REDCAP_DATA_FILE)
