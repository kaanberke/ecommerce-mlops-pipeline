import numpy as np

from model_registry.utils import convert_numpy_types, remove_none_values


def test_convert_numpy_types():
    input_data = {
        "int": np.int32(42),
        "array": np.array([1, 2, 3]),
        "nested": {"ndarray": np.array([4, 5, 6])},
        "list": [np.int64(7), np.float64(8.9)],
    }
    expected_output = {
        "int": 42,
        "array": [1, 2, 3],
        "nested": {"ndarray": [4, 5, 6]},
        "list": [7, 8.9],
    }
    assert convert_numpy_types(input_data) == expected_output


def test_remove_none_values():
    input_data = {"a": 1, "b": None, "c": [1, None, 3], "d": {"x": 10, "y": None}}
    expected_output = {"a": 1, "c": [1, 3], "d": {"x": 10}}
    assert remove_none_values(input_data) == expected_output
