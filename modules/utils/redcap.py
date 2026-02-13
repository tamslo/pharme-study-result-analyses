"""Utils and definitions for REDcap data."""

import json
from pathlib import Path

import requests

from modules.definitions.constants import (
    EHIVE_ID,
    EXTERNAL_DATA_DIRECTORY,
    PHARME_ID,
    get_bool_from_env,
    get_config,
)


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
    use_cache = get_bool_from_env(
        "USE_REDCAP_CACHE",
    )
    redcap_cache_file = Path(
        f"{EXTERNAL_DATA_DIRECTORY}/redcap_users_cache.json",
    )
    if use_cache and Path.exists(redcap_cache_file):
        print("ℹ️ Using cached REDCap users")  # noqa: RUF001, T201
        with Path.open(redcap_cache_file, "r") as user_cache_file:
            user_data = json.load(user_cache_file)
    else:
        user_data = _get_from_redcap("record")
    if use_cache and not Path.exists(redcap_cache_file):
        with Path.open(redcap_cache_file, "w") as user_cache_file:
            json.dump(user_data, user_cache_file)
    return [
        record
        for record in user_data
        if record["study_id"] not in ["JaneDoe", "PharMe_Test"]
    ]
