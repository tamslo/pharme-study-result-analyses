"""Analysis code for investigating wrong comprehension answers."""

import logging

from pandas import DataFrame, Series

from modules.definitions.constants import (
    COMPREHENSION_LOG_FILE,
    MISSING_GENE_QUESTION,
    QUESTION_COLUMN,
    STUDY_GROUP,
)
from modules.definitions.types import StudyGroup, format_time_point_name
from modules.survey_results.comprehension import (
    get_gene_from_question,
    get_medication_from_question,
)
from modules.utils.comparisons import create_comparison_plot
from modules.utils.data import load_data_from_file


def _get_label_definition_for_questions(
    questions: Series,
) -> dict[str, str]:
    label_definition = {}
    for question in sorted(questions, reverse=True):
        label = "NOT HANDLED"
        if question.endswith(":"):
            label = get_gene_from_question(question)
        if question.endswith("at standard dosage?"):
            label = get_medication_from_question(question).capitalize()
        if question.startswith("Which of the following genes"):
            label = MISSING_GENE_QUESTION.capitalize()
        if "should your doctor also consider additional factors" in question:
            label = "Additional factors"
        if "consult your care provider" in question:
            label = "Consult provider"
        label_definition[question] = label
        if label == "NOT HANDLED":
            logger = logging.getLogger(__name__)
            logger.info(
                "Need to define label for '%(question)s' in"
                "_get_label_definition_for_questions",
                {"question": question},
            )

    return label_definition


def analyze_wrong_comprehension_answers(
    wrong_answers: DataFrame,
    time_point_name: str,
) -> None:
    """Plot wrong and missing answers."""
    label_definition = _get_label_definition_for_questions(
        wrong_answers["question"],
    )
    actually_wrong_answers = wrong_answers[wrong_answers["answer"] == False]  # noqa: E712
    not_answered = wrong_answers[wrong_answers["answer"].isna()]
    title = (
        "Wrong answers per question "
        f"({format_time_point_name(time_point_name)})"
    )
    create_comparison_plot(
        title,
        wrong_answers,
        QUESTION_COLUMN,
        y_axis_label="Answers",
        label_definition=label_definition,
        file_name=f"wrong_answers_{time_point_name.lower()}",
        dry_run=False,
        stacked_data={
            "Actually wrong": actually_wrong_answers,
            "Not answered": not_answered,
        },
    )
    wrong_answer_log = load_data_from_file(COMPREHENSION_LOG_FILE)
    wrong_missing_gene_answers = wrong_answer_log[
        wrong_answer_log[QUESTION_COLUMN] == MISSING_GENE_QUESTION
    ]
    wrong_pharme_missing_gene_answers = wrong_missing_gene_answers[
        wrong_missing_gene_answers[STUDY_GROUP] == StudyGroup.PHARME.value
    ]
    missing_gene_wrong_answer_count = len(
        [
            wrong_answer_note
            for wrong_answer_note in wrong_pharme_missing_gene_answers["notes"]
            if not wrong_answer_note.startswith("Overwriting to true")
        ],
    )
    print(  # noqa: T201
        f"Wrong missing gene answers in PharMe arm {()}: "
        f"{missing_gene_wrong_answer_count}",
    )
