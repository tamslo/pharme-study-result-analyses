"""Code to setup the analyses."""

import logging

from modules.session_info import get_manual_file_info, get_run_info
from modules.survey_results.get_case_umars_data import (
    maybe_update_case_umars_data,
)
from modules.survey_results.preprocessing import maybe_preprocess_study_results


def set_up_analysis() -> None:
    """Set up analysis by preprocessing data and checking for updates."""
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.INFO,
        force=True,
    )

    maybe_update_case_umars_data()
    maybe_preprocess_study_results()

    print(f"\n{get_run_info()}\n")  # noqa: T201
    print(get_manual_file_info())  # noqa: T201
