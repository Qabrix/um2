import numpy as np

from src.loader import Loader
from src.cvrp import CVRP

def get_greedy(cvrp: CVRP, starting_localization: int = None) -> np.ndarray:
    localizations_left = list(range(1, cvrp.n_localizations))
    genes = [0]
    capacity_left = cvrp.capacity

    if starting_localization is not None:
        capacity_left -= cvrp.demands[starting_localization]
        genes.append(starting_localization)
        localizations_left.remove(starting_localization)

    while localizations_left:
        i_nearest = localizations_left[
            np.argmin(cvrp.dist_matrix[genes[-1], localizations_left])
        ]
        demand = cvrp.demands[i_nearest]

        if capacity_left - demand >= 0:
            capacity_left -= demand
            genes.append(int(i_nearest))
            localizations_left.remove(int(i_nearest))
        else:
            capacity_left = cvrp.capacity
            genes.append(0)
    
    return np.array([[g for g in genes if g != 0]], dtype=np.int32)


def set_greedy(cvrp: CVRP, starting_localization: int = None):
    cvrp.genes = get_greedy(cvrp, starting_localization)
    cvrp.calc_fit_scores()


def calc_solution(starting_localization: int = None) -> CVRP:
    cvrp = CVRP(
        1,
        Loader.capacity,
        Loader.dist_matrix,
        np.array([local.demand for local in Loader.localizations], dtype=np.int32),
    )
    set_greedy(cvrp, starting_localization)

    return cvrp
