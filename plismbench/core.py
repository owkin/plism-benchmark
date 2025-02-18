"""Core module of :mod:`plismbench`."""


def dummy_function(value: int) -> str:
    """Return string representation of an integer.

    Parameters
    ----------
    value : int
        An integer value.

    Returns
    -------
    str
        The string representation the provided integer.

    Raises
    ------
    TypeError
        If the provided input type is not an integer.
    """
    if type(value) is int:  # pylint: disable=unidiomatic-typecheck
        return str(value)
    raise TypeError(f"Unsupported input type: ({type(value)}). Must be an integer.")
