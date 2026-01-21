"""Code to output general analysis session information."""

from datetime import UTC, datetime
from pathlib import Path

from modules.definitions.constants import (
    MANUAL_FILE_PATHS,
)
from modules.utils.data import (
    get_last_file_modification,
    get_latest_file_modification,
)


def get_run_info() -> str:
    """Get description of current run time."""
    updated = datetime.strftime(
        datetime.now(tz=UTC),
        "%Y-%m-%d %H:%M:%S %Z",
    )
    return f"Last script run: {updated}"


def get_manual_file_info() -> str:
    """Get description of when manual files were updated."""
    message = "The following files may need manual updates:\n"
    time_stamps = {}
    for path_string in MANUAL_FILE_PATHS:
        path = Path(path_string)
        last_modified = get_last_file_modification(path)
        if path.is_dir():
            last_modified = get_latest_file_modification(path.iterdir())
        time_stamps[path.name] = last_modified
    sorted_time_stamps = dict(
        sorted(time_stamps.items(), key=lambda item: item[1]),
    )
    for path, last_modified in sorted_time_stamps.items():
        formatted_last_modified = datetime.fromtimestamp(  # noqa: DTZ006
            last_modified,
        ).strftime(
            "%Y-%m-%d %H:%M",
        )
        is_outdated = (
            datetime.fromtimestamp(  # noqa: DTZ006
                list(sorted_time_stamps.values())[len(sorted_time_stamps) - 1],
            )
            - datetime.fromtimestamp(last_modified)  # noqa: DTZ006
        ).days > 0
        line_ending = ")\n" if not is_outdated else " ⚠️)\n"
        message += (
            f"\t* {path} (last updated {formatted_last_modified}{line_ending}"
        )
    return message
