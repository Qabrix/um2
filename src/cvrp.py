import random

import numpy as np
from numba import boolean, float32, int32, prange
from numba.experimental import jitclass

from src.crosses.cx import cx
from src.crosses.ox import ox
from src.crosses.pmx import pmx


@jitclass(
    [
        ("n_population", int32),
        ("n_localizations", int32),
        ("n_contestants", int32),
        ("capacity", int32),
        ("mut_prob", float32),
        ("cross_prob", float32),
        ("dist_matrix", float32[:, :]),
        ("demands", int32[:]),
        ("genes", int32[:, :]),
        ("fit_scores", float32[:]),
        ("returns", boolean[:, :]),
    ]
)
class CVRP:
    def __init__(
        self,
        n_population: int,
        capacity: int,
        dist_matrix: np.ndarray,
        demands: np.ndarray,
        n_contestants: int = 0,
        mut_prob: float = 0.1,
        cross_prob: float = 0.5,
    ):
        self.n_population = n_population
        self.n_localizations = len(demands)
        self.capacity = capacity
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.genes = np.zeros(
            (self.n_population, self.n_localizations - 1), dtype=np.int32
        )
        self.returns = np.zeros(
            (self.n_population, self.n_localizations - 1), dtype=np.bool8
        )
        self.fit_scores = np.zeros(self.n_population, dtype=np.float32)
        self.n_contestants = n_contestants
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob

        self._init_random_genes()
        self.calc_fit_scores()
    
    def _init_random_genes(self):
        for i_subject in prange(self.n_population):
            self.genes[i_subject] = np.random.permutation(self.n_localizations - 1) + 1

    def calc_fit_scores(self):
        self.fit_scores = np.zeros(self.n_population, dtype=np.float32)

        # iteration over each subject in population
        for i_subject in prange(self.n_population):
            capacity_left = self.capacity

            # getting to the first localization
            i_localization = self.genes[i_subject, 0]
            self.fit_scores[i_subject] += self.dist_matrix[0, i_localization]
            capacity_left -= self.demands[i_localization]

            # getting to next localizations
            for step in range(self.n_localizations - 2):
                i_localization = self.genes[i_subject, step]
                i_next_localization = self.genes[i_subject, step + 1]
                demand = self.demands[i_next_localization]

                # check if we have to return to the magazine
                if capacity_left - demand >= 0 and not self.returns[i_subject, step]:
                    self.fit_scores[i_subject] += self.dist_matrix[
                        i_localization, i_next_localization
                    ]
                else:
                    capacity_left = self.capacity
                    self.fit_scores[i_subject] += self.dist_matrix[i_localization, 0]
                    self.fit_scores[i_subject] += self.dist_matrix[
                        0, i_next_localization
                    ]

                capacity_left -= demand

            # getting back to the magazine
            i_localization = self.genes[i_subject, -1]
            self.fit_scores[i_subject] += self.dist_matrix[i_localization, 0]
            capacity_left -= self.demands[i_localization]

    def _tournament_selection(self) -> tuple[np.ndarray, int]:
        random_indices = np.random.choice(
            self.n_population, size=self.n_contestants, replace=False
        )
        random_fit_scores = self.fit_scores[random_indices]
        i_best = random_indices[np.argmin(random_fit_scores)]
        return self.genes[i_best], i_best

    def mutate_gene(self, i_subject: int):
        gene = self.genes[i_subject]

        # flip
        if np.random.random() < self.mut_prob / 4:
            flip_indices = np.random.choice(len(gene), 2, replace=False)
            gene[flip_indices] = np.flip(gene[flip_indices])

        # random returns
        if np.random.random() < self.mut_prob / 5:
            random_indices = np.random.choice(
                len(gene), random.randrange(3), replace=False
            )
            returns = self.returns[i_subject]
            returns[random_indices] = np.invert(returns[random_indices])

        # swap
        if np.random.random() < self.mut_prob:
            swap_indices = np.random.choice(len(gene), 2, replace=False)
            gene[swap_indices] = gene[swap_indices[::-1]]

        # partial shuffle
        if np.random.random() < self.mut_prob / 10:
            shuffle_indices = np.random.choice(
                len(gene),
                random.randrange(self.n_localizations - 1) // 2,
                replace=False,
            )
            np.random.shuffle(gene[shuffle_indices])

        # whole shuffle
        if np.random.random() < self.mut_prob / 20:
            gene = np.random.permutation(self.n_localizations - 1) + 1

    def cross(self, i_subject: int):
        parent1, i_parent1 = self._tournament_selection()
        if np.random.random() < self.cross_prob:
            parent2, i_parent2 = self._tournament_selection()

            returns1 = self.returns[i_parent1]
            returns2 = self.returns[i_parent2]

            i_random = random.randrange(2)
            if i_random == 0:
                self.genes[i_subject], self.returns[i_subject] = cx(
                    parent1, parent2, returns1, returns2
                )
            elif i_random == 1:
                self.genes[i_subject], self.returns[i_subject] = ox(
                    parent1, parent2, returns1, returns2
                )
            # elif i_random == 2:
            #     self.genes[i_subject] = pmx(parent1, parent2)
            #     self.returns[i_subject] = np.zeros_like(self.returns[i_subject])
        else:
            self.genes[i_subject] = np.copy(parent1)

    def best_fit_score(self) -> tuple[np.ndarray, np.float32, np.ndarray]:
        best_index = int(np.argmin(self.fit_scores))
        return self.genes[best_index], self.fit_scores[best_index], self.returns[best_index],
