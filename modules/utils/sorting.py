"""Sort data by labels."""

from collections import OrderedDict

from pandas import Series

from modules.definitions.constants import MULTIPLE_VALUES_SEPARATOR
from modules.utils.data import value_is_nan

SORT_LAST_INDEX = 10000
NOT_ANSWERED_LABEL = "Not answered"
NOT_ANSWERED_SORT_INDEX = SORT_LAST_INDEX + 1


def _get_single_sort_index(value: str, label_definition: OrderedDict) -> int:
    if value in label_definition:
        return list(label_definition.keys()).index(value)
    if value == NOT_ANSWERED_LABEL:
        return NOT_ANSWERED_SORT_INDEX
    return SORT_LAST_INDEX


def _get_sort_index(value: str, label_definition: OrderedDict) -> float:
    if value_is_nan(value):
        return -1
    if MULTIPLE_VALUES_SEPARATOR in value:
        single_values = value.split(MULTIPLE_VALUES_SEPARATOR)
        accumulated_sort_index = 0
        for index, single_value in enumerate(single_values):
            factor = 1 if index == 0 else 0.1
            accumulated_sort_index += factor * _get_single_sort_index(
                single_value,
                label_definition,
            )
        return accumulated_sort_index
    return _get_single_sort_index(value, label_definition)


def sort_by_label(values: Series, label_definition: OrderedDict) -> Series:
    """Sort series index or values by label definition."""
    return [_get_sort_index(value, label_definition) for value in values]
