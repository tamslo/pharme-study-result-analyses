"""Utils and definitions for REDcap data."""

import requests

from modules.definitions.constants import EHIVE_ID, PHARME_ID, get_config


def _get_from_redcap(content: str) -> list[dict]:
    config = get_config()
    redcap_api_url = config["REDCAP_API_URL"]
    redcap_api_token = config["REDCAP_API_KEY"]
    return requests.post(
        redcap_api_url,
        data={
            "token": redcap_api_token,
            "content": content,
            "format": "json",
        },
        timeout=3,
    ).json()


def get_pharme_id(
    redcap_users: list[dict],
    ehive_id: str,
) -> str:
    """Get PharMe ID from REDCap user with ehive ID."""
    redcap_user = next(
        user for user in redcap_users if user[EHIVE_ID] == ehive_id
    )
    return redcap_user[PHARME_ID]


def get_redcap_users() -> list[dict]:
    """Get a the list of participants (records) in REDcap."""
    return [
        record
        for record in _get_from_redcap("record")
        if record["study_id"] not in ["JaneDoe", "PharMe_Test"]
    ]
