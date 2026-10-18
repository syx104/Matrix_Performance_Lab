import numpy as np
import pytest

from matmul import naive_matmul

def test_small_matrix():
    a=np.array([[1,2],[3,4]])
    b=np.array([[5,6],[7,8]])

    expected=np.array([[19,22],[43,50]])
    actual=naive_matmul(a,b)

    np.testing.assert_allclose(actual,expected)

def test_rectangular_matrix():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [5, 6]])

    actual = naive_matmul(a, b)
    expected = a @ b

    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, expected)

def test_random_matrix():
    rng = np.random.default_rng(2026)

    a = rng.standard_normal((5, 7))
    b = rng.standard_normal((7, 3))

    actual = naive_matmul(a, b)
    expected = a @ b

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_invalid_shape():
    a = np.ones((2, 3))
    b = np.ones((4, 2))

    with pytest.raises(ValueError):
        naive_matmul(a, b)


def test_empty_output():
    a = np.empty((0, 3))
    b = np.ones((3, 2))

    actual = naive_matmul(a, b)

    assert actual.shape == (0, 2)
    np.testing.assert_allclose(actual, a @ b)