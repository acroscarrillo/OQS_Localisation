"""Functions for the calculation of moments and generalized fractal dimensions."""

import math
from numba import njit, vectorize
import numpy as np
from nptyping import NDArray, Complex, Float, Shape

@njit
def _calculate_moment(
    state: NDArray[Shape['*'], Complex],
    width: int,
    power: int | float,
) -> float:
    moment = 0
    for window_index in range(len(state) // width):
        bin_probability = 0
        for window_el in state[window_index*width : window_index*width + width]:
            bin_probability += window_el.real**2 + window_el.imag**2
        moment += bin_probability**power
    return moment

@njit
def calculate_moments_mutating(
    out: NDArray[Shape['N'], Complex],
    states: NDArray[Shape['N, *'], Complex],
    width: int,
    power: int | float,
) -> None:
    """Calculates the moments of an array of states.

    Calculates the moment of a probability distribution for the system residing in a set
    of non-overlapping boxes. Stores the result in the first argument.

    Args:
        states (2D array of complex128): An array of quantum states, with each
            state indexed by the first dimension.
        width: The width of each box.
        power: The moment of the distribution calculated.

    Returns:
        None.

    Raises:
        ValueError: if states is not a 1D or 2D array, the state length is not an
            integer multiple of the width, or if out and states do not have the same
            first dimension length.
    """
    if states.shape[0] != out.shape[0]:
        raise ValueError("states and out must have the same first dimension length.")
    if states.shape[-1] % width == 0:
        if states.ndim == 2:
            for (index, _) in enumerate(out):
                out[index] = _calculate_moment(states[index,:], width, power)
            return None
        raise ValueError("states must be a 2D array.")
    raise ValueError("State length is not an integer multiple of width.")

@njit
def calculate_moments(
    states: NDArray[Shape['N, *'], Complex],
    width: int,
    power: int | float,
) -> float | NDArray[Shape['N'], Float]:
    """Calculates the moment(s) of a state or array of states.

    Calculates the moment of a probability distribution for the system residing in a set
    of non-overlapping boxes.

    Args:
        states (1D or 2D array of complex128): An array of quantum states, with each
            state indexed by the first dimension.
        width: The width of each box.
        power: The moment of the distribution calculated.

    Returns:
        A single float or 1D array of floats for the moments.

    Raises:
        ValueError: if states is not a 1D or 2D array, or the state length is not an
            integer multiple of the width.
    """
    if states.shape[-1] % width == 0:
        if states.ndim == 1:
            return _calculate_moment(states, width, power)
        if states.ndim == 2:
            out = np.zeros(states.shape[0])
            calculate_moments_mutating(out, states, width, power)
            return out
        raise ValueError("states must be a 1D or 2D array.")
    raise ValueError("State length is not an integer multiple of width.")

@vectorize
def mass_exponent(
    moment: float,
    width: int,
    length: int,
) -> float:
    """Calculates the mass exponents of an array of moments element wise."""
    return math.log(moment) / math.log(width/length)

@vectorize
def generalized_fractal_dimension(
    moment: float,
    width: int,
    power: int | float,
    length: int,
) -> float:
    """Calculates the generalized fractal dimensions of an array of moments element wise.
    """
    return mass_exponent(moment, width, length) / (power - 1)
