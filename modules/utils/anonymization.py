"""Code for survey anonymization."""

import uuid
from pathlib import Path

from pandas import DataFrame

from modules.definitions.constants import (
    EHIVE_ID,
    PARTICIPANT_ID,
    PARTICIPANT_ID_MAP_FILE,
)
from modules.definitions.types import ThisShouldNeverHappenError
from modules.utils.data import load_data_from_file, write_data_frame


def get_participant_id_map() -> dict[str, str]:
    """Read the stored map of ehive and participant IDs."""
    participant_id_map_file = Path(PARTICIPANT_ID_MAP_FILE)
    if not participant_id_map_file.exists():
        return {}
    participant_id_map_data = load_data_from_file(PARTICIPANT_ID_MAP_FILE)
    participant_id_map = {}
    for _, id_mapping in participant_id_map_data.iterrows():
        ehive_id = id_mapping[EHIVE_ID]
        participant_id = id_mapping[PARTICIPANT_ID]
        participant_id_map[ehive_id] = participant_id
    return participant_id_map


def reveal_ehive_id(
    participant_map: dict[str, str],
    participant_id: str,
) -> str:
    """Get the ehive ID from the anonymous participant ID. Use with caution."""
    inverted_participant_map = {
        participant_id: ehive_id
        for ehive_id, participant_id in participant_map.items()
    }
    return inverted_participant_map[participant_id]


def _get_new_id(current_ids: list[str]) -> str:
    new_id = str(uuid.uuid4())
    while new_id in current_ids:
        new_id = str(uuid.uuid4())
    return new_id


def _save_user_map(user_id_map: dict[str, str]) -> None:
    participant_id_map_data = []
    for ehive_id, participant_id in user_id_map.items():
        participant_id_map_data.append(
            {
                EHIVE_ID: ehive_id,
                PARTICIPANT_ID: participant_id,
            },
        )
    write_data_frame(
        DataFrame.from_dict(participant_id_map_data),
        PARTICIPANT_ID_MAP_FILE,
    )


def anonymize_results(survey_results: DataFrame) -> DataFrame:
    """Anonymizes survey results with user ID map."""
    participant_id_map = get_participant_id_map()
    for index, result in survey_results.iterrows():
        ehive_id = result[PARTICIPANT_ID]
        if ehive_id in participant_id_map.values():
            error_message = (
                "🚨 Attempting to anonymize an already anonymized ID!"
            )
            raise ThisShouldNeverHappenError(error_message)
        if ehive_id in participant_id_map:
            anonymous_id = participant_id_map[ehive_id]
        else:
            anonymous_id = _get_new_id(current_ids=participant_id_map.values())
            participant_id_map[ehive_id] = anonymous_id
        survey_results.loc[index, PARTICIPANT_ID] = anonymous_id
    _save_user_map(participant_id_map)
    return survey_results
