"""Unit tests for :mod:`plismbench.core`."""

import pytest

from plismbench.core import dummy_function


@pytest.mark.parametrize(
    "value, result",
    [
        (2, "2"),
        (10, "10"),
        (-4, "-4"),
        (0, "0"),
    ],
)
def test_dummy_function(value, result):
    """Test main function of :func:`plismbench.core.dummy_function`."""
    assert dummy_function(value) == result


@pytest.mark.parametrize("value", [2.0, "foo", False, None])
def test_dummy_function_wrong_input_type(value):
    """Test wrong input type to :func:`plismbench.core.dummy_function`."""
    expected_exception_message = (
        r"Unsupported input type: \(" f"{type(value)}" r"\). Must be an integer."
    )
    with pytest.raises(TypeError, match=expected_exception_message):
        _ = dummy_function(value)
