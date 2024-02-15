from enum import Enum

import numpy as np


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return np.linalg.norm(x - y)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - (dot_product(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y)


class Distance(Enum):
    EUCLIDEAN = euclidean
    COSINE = cosine
    DOT_PRODUCT = dot_product
