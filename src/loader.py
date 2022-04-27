from dataclasses import dataclass

import numpy as np
import scipy.spatial


@dataclass
class Localization:
    coords: tuple[int]
    demand: int


class LengthMetaclass(type):
    def __len__(self):
        return self.localizations_size()


class Loader(object, metaclass=LengthMetaclass):
    capacity: int
    dimension: int
    dist_matrix: np.ndarray
    localizations: list[Localization]

    @staticmethod
    def set_capacity(capacity: str):
        Loader.capacity = int(capacity)

    @staticmethod
    def set_dimension(dimension: str):
        Loader.dimension = int(dimension)

    @classmethod
    def localizations_size(cls):
        return len(cls.localizations)

    @staticmethod
    def from_path(path):
        Loader.localizations = []

        fields = {"DIMENSION": Loader.set_dimension, "CAPACITY": Loader.set_capacity}

        def _assign_demand_action(line: str):
            idx, demand = list(
                map(int, filter(lambda x: len(x) > 0, line[:-1].split(" ")))
            )
            Loader.localizations[idx - 1].demand = demand

            return _assign_demand_action if idx < Loader.dimension else None

        def _assign_localizations_action(line: str):
            if line.startswith("DEMAND_SECTION"):
                return _assign_demand_action

            Loader.localizations.append(
                Localization(coords=tuple(map(int, line.split(" ")[1:])), demand=0)
            )

            return _assign_localizations_action

        def _assign_starters_action(line: str):
            if line.startswith("NODE_COORD_SECTION"):
                return _assign_localizations_action

            field, value = line.split(" : ")
            fields.get(field, lambda _: None)(value)

            return _assign_starters_action

        current_action = _assign_starters_action
        with path.open("r", encoding="utf8") as file:
            for line in file:
                if current_action is None:
                    break

                current_action = current_action(line)

            all_coords = [local.coords for local in Loader.localizations]
            Loader.dist_matrix = scipy.spatial.distance_matrix(
                all_coords, all_coords
            ).astype(np.float32)
