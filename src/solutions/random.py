import numpy as np

from src.loader import Loader
from src.cvrp import CVRP


def calc_solution() -> CVRP:
    cvrp = CVRP(
        1,
        Loader.capacity,
        Loader.dist_matrix,
        np.array([local.demand for local in Loader.localizations], dtype=np.int32),
    )

    return cvrp
