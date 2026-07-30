#!/usr/bin/env python3
"""CPU reference check for the cached distributed-TDMA factor formulas."""

from __future__ import annotations

import numpy as np


def prepare_local(a, b, c):
    a, b, c = a.copy(), b.copy(), c.copy()
    raw_lower, raw_upper = a.copy(), c.copy()
    n = b.size
    lower = np.zeros(n)
    upper = np.zeros(n)

    inverse = 1.0 / b[0]
    a0, c0 = a[0] * inverse, c[0] * inverse
    a[0], b[0], c[0], upper[0] = a0, inverse, c0, c0

    inverse = 1.0 / b[1]
    a1, c1 = a[1] * inverse, c[1] * inverse
    a[1], b[1], c[1], upper[1] = a1, inverse, c1, c1

    for row in range(2, n):
        a0, c0 = a1, c1
        raw_a = a[row]
        inverse = 1.0 / (b[row] - raw_a * c0)
        lower[row] = raw_a * inverse
        a1 = -lower[row] * a0
        c1 = c[row] * inverse
        a[row], b[row], c[row], upper[row] = a1, inverse, c1, c1

    reduced_a = np.array([0.0, a1])
    reduced_b = np.ones(2)
    reduced_c = np.array([0.0, c1])

    a1, c1 = a0, c0
    for row in range(n - 3, 0, -1):
        a0 = a[row] - c[row] * a1
        c0 = -c[row] * c1
        a[row], c[row], a1, c1 = a0, c0, a0, c0

    a0, c0 = a[0], c[0]
    lower[0] = 1.0 / (1.0 - a1 * c0)
    a0 = lower[0] * a0
    c0 = -lower[0] * c0 * c1
    a[0], c[0] = a0, c0
    reduced_a[0], reduced_c[0] = a0, c0

    # The CUDA implementation stores these factors compactly as two z-row
    # vectors plus inverse pivots, not as two full 3D arrays.
    np.testing.assert_allclose(lower[2:], raw_lower[2:] * b[2:])
    np.testing.assert_allclose(upper[1:], raw_upper[1:] * b[1:])

    return a, b, c, lower, upper, reduced_a, reduced_b, reduced_c


def apply_local_compact(
    inverse,
    lower_row,
    upper_row,
    row1_inverse,
    row1_upper,
    rhs,
):
    rhs = rhs.copy()
    n = rhs.size
    rhs[0] *= inverse[0]
    rhs[1] *= inverse[1]
    for row in range(2, n):
        rhs[row] = inverse[row] * (
            rhs[row] - lower_row[row] * rhs[row - 1]
        )

    reduced_rhs = np.array([0.0, rhs[-1]])
    for row in range(n - 3, 0, -1):
        rhs[row] -= upper_row[row] * inverse[row] * rhs[row + 1]
    rhs[0] = row1_inverse * (rhs[0] - row1_upper * rhs[1])
    reduced_rhs[0] = rhs[0]
    return rhs, reduced_rhs


def prepare_thomas(a, b, c):
    a, b, c = a.copy(), b.copy(), c.copy()
    inverse = 1.0 / b[0]
    b[0], c[0] = inverse, c[0] * inverse
    previous_c = c[0]
    for row in range(1, b.size):
        raw_a = a[row]
        inverse = 1.0 / (b[row] - raw_a * previous_c)
        a[row], b[row], c[row] = (
            raw_a * inverse,
            inverse,
            c[row] * inverse,
        )
        previous_c = c[row]
    return a, b, c


def apply_thomas(a, inverse, c, rhs):
    rhs = rhs.copy()
    rhs[0] *= inverse[0]
    for row in range(1, rhs.size):
        rhs[row] = inverse[row] * rhs[row] - a[row] * rhs[row - 1]
    for row in range(rhs.size - 2, -1, -1):
        rhs[row] -= c[row] * rhs[row + 1]
    return rhs


def check(nranks, trials=25):
    rng = np.random.default_rng(20260730 + nranks)
    nlocal = 7
    nrows = nranks * nlocal
    lower = -0.2 - 0.3 * rng.random(nrows)
    upper = -0.2 - 0.3 * rng.random(nrows)
    diagonal = 2.5 + rng.random(nrows)
    lower[0] = 0.0
    upper[0] = 0.0
    upper[-1] = 0.0
    diagonal[0] = 1.0

    local_factors = []
    reduced_a, reduced_b, reduced_c = [], [], []
    for rank in range(nranks):
        section = slice(rank * nlocal, (rank + 1) * nlocal)
        factors = prepare_local(
            lower[section], diagonal[section], upper[section]
        )
        local_factors.append(factors[:5])
        reduced_a.extend(factors[5])
        reduced_b.extend(factors[6])
        reduced_c.extend(factors[7])

    reduced_factors = prepare_thomas(
        np.asarray(reduced_a),
        np.asarray(reduced_b),
        np.asarray(reduced_c),
    )
    matrix = (
        np.diag(diagonal)
        + np.diag(lower[1:], -1)
        + np.diag(upper[:-1], 1)
    )

    max_error = 0.0
    for _ in range(trials):
        rhs = rng.normal(size=nrows)
        partial, reduced_rhs = [], []
        for rank, factors in enumerate(local_factors):
            section = slice(rank * nlocal, (rank + 1) * nlocal)
            local_rhs, endpoints = apply_local_compact(
                factors[1],
                lower[section],
                upper[section],
                factors[3][0],
                factors[4][0],
                rhs[section],
            )
            partial.append(local_rhs)
            reduced_rhs.extend(endpoints)

        boundary = apply_thomas(
            *reduced_factors, np.asarray(reduced_rhs)
        )
        solution = []
        for rank, (local_rhs, factors) in enumerate(
            zip(partial, local_factors)
        ):
            start, end = boundary[2 * rank : 2 * rank + 2]
            local_rhs[0], local_rhs[-1] = start, end
            for row in range(1, nlocal - 1):
                local_rhs[row] -= (
                    factors[0][row] * start + factors[2][row] * end
                )
            solution.extend(local_rhs)

        reference = np.linalg.solve(matrix, rhs)
        max_error = max(
            max_error,
            float(np.max(np.abs(np.asarray(solution) - reference))),
        )

    assert max_error < 2.0e-12
    print(
        f"[PASS] nranks={nranks}, RHS={trials}, "
        f"max-error={max_error:.3e}"
    )


if __name__ == "__main__":
    check(2)
    check(4)
