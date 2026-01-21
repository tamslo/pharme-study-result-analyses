"""Data preprocessing for comprehension data."""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pandas import DataFrame

from modules.definitions.constants import (
    ALSO_PHENOTYPE_ADAPTIONS_COMMUNICATED,
    ANSWER_COLUMN,
    COMPREHENSION_DATA,
    COMPREHENSION_LOG_FILE,
    DPYD_INDETERMINATE_CASTS,
    MISSING_GENE_QUESTION,
    ONLY_GENOTYPE_ADAPTIONS_COMMUNICATED,
    PARTICIPANT_ID,
    QUESTION_COLUMN,
    SECRET_COMPREHENSION_LOG_FILE,
    STUDY_GROUP,
)
from modules.definitions.types import Survey
from modules.survey_results.get_data import get_survey_results
from modules.survey_results.normalize import (
    get_normalized_survey_data,
    replace_in_columns,
)
from modules.survey_results.redcap_data import get_study_group
from modules.utils.anonymization import get_participant_id_map, reveal_ehive_id
from modules.utils.data import (
    get_data_path,
    value_is_nan,
    write_data_frame,
)
from modules.utils.redcap import get_pharme_id, get_redcap_users

REMOVE_COMPREHENSION_COLUMN_FORMULATIONS = [
    " in the counseling session",
    " in the PharMe app",
    " from the PGx report available in MyChart",
    " from the PharMe app",
]


def get_gene_from_question(question: str) -> str:
    """Get gene name from comprehension question."""
    return question.split(" ")[-1].replace(":", "")


def get_medication_from_question(question: str) -> str:
    """Get medication name from comprehension question."""
    return question.replace(
        (
            "According to your PGx test result, if you ever needed to take the "
            "medication "
        ),
        "",
    ).replace(", could you take it at standard dosage?", "")


def _write_to_log(log_file, row: list) -> None:  # noqa: ANN001
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(row)


def _write_log_headers() -> None:
    with Path.open(COMPREHENSION_LOG_FILE, "w") as log_file:
        _write_to_log(
            log_file,
            ["timestamp", STUDY_GROUP, QUESTION_COLUMN, ANSWER_COLUMN, "notes"],
        )
    with Path.open(SECRET_COMPREHENSION_LOG_FILE, "w") as secret_log_file:
        _write_to_log(secret_log_file, [PARTICIPANT_ID])


def _write_preprocessing_log(
    question: str,
    answer: str,
    notes: str,
    participant_id: str,
) -> None:
    timestamp = datetime.strftime(
        datetime.now(tz=UTC),
        "%Y-%m-%d %H:%M:%S %Z",
    )
    study_group = get_study_group(participant_id).value
    with Path.open(COMPREHENSION_LOG_FILE, "a") as log_file:
        _write_to_log(
            log_file,
            [timestamp, study_group, question, answer, notes],
        )
    with Path.open(SECRET_COMPREHENSION_LOG_FILE, "a") as secret_log_file:
        _write_to_log(
            secret_log_file,
            [participant_id],
        )


def _get_static_answer(participant_answer: str) -> bool:
    static_answer = "yes"
    return participant_answer == static_answer


def _analyze_missing_gene(
    participant_id: str,
    participant_answer: str,
    participant_genes: dict,
    pharme_id: str,
) -> bool:
    actual_missing_gene = "MTHFR"
    actual_result_correct = participant_answer == actual_missing_gene
    if actual_result_correct:
        return True
    participant_answer_phenotype = participant_genes[participant_answer][
        "phenotype"
    ].lower()
    # Not missing in list but could still be counted as missing if
    # indeterminate
    overwrite_result = participant_answer_phenotype == "indeterminate"
    if overwrite_result:
        _write_preprocessing_log(
            MISSING_GENE_QUESTION,
            participant_answer,
            "Overwriting to true because Indeterminate",
            participant_id,
        )
        return True
    if participant_answer == "DPYD" and pharme_id in DPYD_INDETERMINATE_CASTS:
        _write_preprocessing_log(
            MISSING_GENE_QUESTION,
            participant_answer,
            (
                "Overwriting to true because Indeterminate (incomplete "
                "preprocessing)"
            ),
            participant_id,
        )
        return True
    wrong_answer_message = (
        f"{participant_answer} "
        f"{participant_genes[participant_answer]['phenotype']}"
    )
    result_adaptions_communicated = {
        **ONLY_GENOTYPE_ADAPTIONS_COMMUNICATED,
        **ALSO_PHENOTYPE_ADAPTIONS_COMMUNICATED,
    }
    if (
        pharme_id in result_adaptions_communicated
        and participant_answer == result_adaptions_communicated[pharme_id]
    ):
        wrong_answer_message += (
            "; ℹ️ participant selected a gene that was simplified in PharMe"  # noqa: RUF001
        )
    _write_preprocessing_log(
        MISSING_GENE_QUESTION,
        participant_answer,
        wrong_answer_message,
        participant_id,
    )
    return False


def _analyze_phenotype_answer(
    participant_id: str,
    column: str,
    participant_answer: str,
    participant_genes: dict,
) -> bool:
    gene = get_gene_from_question(column)
    actual_phenotype = (
        participant_genes[gene]["phenotype"].split(" ")[0].lower()
    )
    answer_correct = participant_answer == actual_phenotype
    if not answer_correct:
        wrong_answer_message = f"{gene} {participant_genes[gene]['phenotype']}"
        if (
            participant_id in ALSO_PHENOTYPE_ADAPTIONS_COMMUNICATED
            and ALSO_PHENOTYPE_ADAPTIONS_COMMUNICATED[participant_genes] == gene
        ):
            wrong_answer_message += (
                "; phenotype for participant was adapted, 👀 check if wrong "
                "answer included there"
            )
        _write_preprocessing_log(
            gene,
            participant_answer,
            wrong_answer_message,
            participant_id,
        )
    return answer_correct


def _analyze_medication_answer(
    participant_id: str,
    column: str,
    participant_answer: str,
    participant_data: dict,
) -> bool:
    medication_at_standard_dose_phenotypes = {
        "ibuprofen": {
            "gene": "CYP2C9",
            "phenotypes": [
                "Normal Metabolizer",
                "Intermediate Metabolizer",
                "Indeterminate",
            ],
        },
        "simvastatin": {
            "gene": "SLCO1B1",
            "phenotypes": [
                "Increased Function",
                "Normal Function",
                "Indeterminate",
            ],
        },
        "citalopram": {
            "gene": "CYP2C19",
            "phenotypes": [
                "Rapid Metabolizer",
                "Normal Metabolizer",
                "Intermediate Metabolizer",
                "Indeterminate",
            ],
        },
        "clopidogrel": {
            "gene": "CYP2C19",
            "phenotypes": [
                "Indeterminate",
                "Ultrarapid Metabolizer",
                "Rapid Metabolizer",
                "Normal Metabolizer",
            ],
        },
    }
    medication = get_medication_from_question(column)
    medication_standard_dose_data = medication_at_standard_dose_phenotypes[
        medication
    ]
    medication_gene = medication_standard_dose_data["gene"]
    standard_dose_phenotypes = medication_standard_dose_data["phenotypes"]
    participant_genes = participant_data["genes"]
    participant_phenotype = participant_genes[medication_gene]["phenotype"]
    correct_answer = (
        "yes" if participant_phenotype in standard_dose_phenotypes else "no"
    )
    answer_correct = participant_answer == correct_answer
    if not answer_correct:
        participant_genotype = participant_genes[medication_gene]["genotype"]
        participant_result_details = participant_phenotype
        if medication_gene == "CYP2C9":
            cyp2c9_activity_scores = {
                "*1/*1": 2.0,
                "*1/*2": 1.5,
                "*1/*3": 1.0,
                "*1/*11": 1.5,
                "*2/*2": 1.0,
                "*2/*3": 0.5,
                "*3/*3": 0.0,
            }
            if participant_genotype not in cyp2c9_activity_scores:
                error_message = (
                    "Need to extend cyp2c9_activity_scores by "
                    f"{participant_genotype}"
                )
                raise Exception(error_message)  # noqa: TRY002
            participant_result_details += (
                f" ({cyp2c9_activity_scores[participant_genotype]})"
            )
        wrong_answer_message = f"{medication_gene} {participant_result_details}"
        participant_on_medication = json.loads(
            participant_data["medications"][medication],
        )
        if participant_on_medication:
            wrong_answer_message += (
                "; 🚨 patient is taking medication, check if counseling needed"
            )
        _write_preprocessing_log(
            medication,
            participant_answer,
            wrong_answer_message,
            participant_id,
        )
    return answer_correct


def _get_comprehension_result(
    participant_id: str,
    pharme_id: str,
    participant_answer: str,
    participant_data: dict,
    column: str,
) -> bool:
    comprehension_columns = _get_comprehension_columns()
    static_answer_columns = [
        comprehension_columns[4],
        comprehension_columns[5],
    ]
    missing_gene_column = comprehension_columns[6]
    phenotype_answer_columns = [
        comprehension_columns[7],
        comprehension_columns[9],
        comprehension_columns[11],
    ]
    medication_answer_columns = [
        comprehension_columns[8],
        comprehension_columns[10],
        comprehension_columns[12],
        comprehension_columns[13],
    ]
    if column in static_answer_columns:
        return _get_static_answer(participant_answer)
    if column == missing_gene_column:
        return _analyze_missing_gene(
            participant_id,
            participant_answer,
            participant_data["genes"],
            pharme_id,
        )
    if column in phenotype_answer_columns:
        return _analyze_phenotype_answer(
            participant_id,
            column,
            participant_answer,
            participant_data["genes"],
        )
    if column in medication_answer_columns:
        return _analyze_medication_answer(
            participant_id,
            column,
            participant_answer,
            participant_data,
        )
    return participant_answer


def _get_comprehension_columns() -> list[str]:
    return replace_in_columns(
        Survey.COMPREHENSION_APP,
        REMOVE_COMPREHENSION_COLUMN_FORMULATIONS,
        "",
    )


def _get_normalized_survey_results() -> DataFrame:
    normalized_results = get_normalized_survey_data(
        Survey.COMPREHENSION_COUNSELING,
        Survey.COMPREHENSION_APP,
        Survey.COMPREHENSION,
        REMOVE_COMPREHENSION_COLUMN_FORMULATIONS,
        "",
    )
    return normalized_results.drop("score", axis=1)


def maybe_map_comprehension_data() -> DataFrame:
    """Map from comprehension data to binary comprehension results."""
    explicit_comprehension_results = _get_normalized_survey_results()
    logger = logging.getLogger(__name__)
    anonymous_comprehension_results_path = get_data_path(Survey.COMPREHENSION)
    if anonymous_comprehension_results_path.exists():
        present_comprehension_results = get_survey_results(Survey.COMPREHENSION)
        if len(present_comprehension_results.index) == len(
            explicit_comprehension_results.index,
        ):
            logger.info(
                "Loaded preprocessed comprehension data from file, "
                "all comprehension data from surveys present",
            )
            return
    logger.info("Preprocessing comprehension data...")
    _write_log_headers()
    comprehension_columns = _get_comprehension_columns()
    anonymous_comprehension_results = DataFrame(
        columns=comprehension_columns,
    )
    redcap_users = get_redcap_users()
    participant_id_map = get_participant_id_map()
    with Path.open(COMPREHENSION_DATA) as comprehension_data_file:
        comprehension_data = json.load(comprehension_data_file)
    missing_comprehension_data = []
    for index, participant_result in explicit_comprehension_results.iterrows():
        participant_id = participant_result[PARTICIPANT_ID]
        pharme_id = get_pharme_id(
            redcap_users,
            reveal_ehive_id(participant_id_map, participant_id),
        )
        if pharme_id not in comprehension_data:
            missing_comprehension_data.append(pharme_id)
            continue
        participant_data = comprehension_data[pharme_id]
        for column in participant_result.index:
            if value_is_nan(participant_result[column]):
                continue
            participant_result[column] = _get_comprehension_result(
                participant_id,
                pharme_id,
                participant_result[column],
                participant_data,
                column,
            )
        anonymous_comprehension_results.loc[index] = participant_result
    missing_comprehension_data = set(missing_comprehension_data)
    if len(missing_comprehension_data) > 0:
        logger.warning(
            "⚠️ No comprehension data for PharMe user(s) %(pharme_ids)s; "
            "please update %(comprehension_data_path)s",
            {
                "pharme_ids": ", ".join(missing_comprehension_data),
                "comprehension_data_path": COMPREHENSION_DATA,
            },
        )
    write_data_frame(
        anonymous_comprehension_results,
        anonymous_comprehension_results_path,
    )
