"""Code to analyze feelings."""

FACTOR_SUBSCALES = {
    "negative emotions": [
        "How upset did you feel about your PGx test result?",
        "How anxious or nervous did you feel about your PGx test result?",
        "How sad did you feel about your PGx test result?",
    ],
    "positive feelings": [
        "How happy did you feel about your PGx test result?",
        "How relieved did you feel about your PGx test result?",
        "How much did you feel that you understood clearly your choices for "
        "prevention or early detection of side effects?",
        "How helpful was the information you received from your PGx test "
        "result in planning for the future?",
    ],
    "uncertainty": [
        "How frustrated did you feel that there are only PGx recommendations "
        "for about 100 medications (out of thousands of existing medications)?",
        "How uncertain did you feel about what your PGx test result means "
        "for you?",
        "How uncertain did you feel about what your PGx test result means for "
        "your child(ren) and/or family's response to medications?",
    ],
    "privacy concerns": [
        "How concerned did you feel that your PGx test result would affect "
        "your health insurance status?",
        "How concerned did you feel that your PGx test result would affect "
        "your employment status?",
    ],
}
