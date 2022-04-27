from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm import tqdm

from src.loader import Loader
from src.cvrp import CVRP


def calc_solution(
    n_population: int,
    n_iters: int,
    n_contestants: int,
    cross_prob: float,
    mut_prob: float,
    init_genes: np.ndarray = None
) -> tuple[CVRP, pd.DataFrame]:
    cvrp = CVRP(
        n_population=n_population,
        capacity=Loader.capacity,
        dist_matrix=Loader.dist_matrix,
        demands=np.array(
            [local.demand for local in Loader.localizations], dtype=np.int32
        ),
        n_contestants=n_contestants,
        cross_prob=cross_prob,
        mut_prob=mut_prob,
    )

    if init_genes is not None:
        cvrp.genes[0] = init_genes[0]
        cvrp.calc_fit_scores()

    scores = []
    pbar = tqdm(range(n_iters))
    for i in pbar:
        perform_ga_iteration(cvrp)
        cvrp.calc_fit_scores()

        best_fit_score = cvrp.best_fit_score()[1]
        pbar.set_description(f"i {i}: fit_score: {best_fit_score}")
        scores.append((i, best_fit_score))

    return cvrp, pd.DataFrame.from_records(scores, columns=["i", "fit_score"])


@njit
def perform_ga_iteration(cvrp):
    for i_subject in prange(cvrp.n_population):
        cvrp.cross(i_subject)
        cvrp.mutate_gene(i_subject)
