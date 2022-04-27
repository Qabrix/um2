from routes import ROUTES_PATH
from src.loader import Loader
from src.solutions.genethic import calc_solution as genethic_solution
from src.solutions.greedy import calc_solution as greedy_solution
from src.solutions.random import calc_solution as random_solution


def main():
    Loader.from_path(ROUTES_PATH / "tai385.vrp")
    # greedy_solution()
    # random_solution()
    genethic_solution(1000, 2000, 50, 0.3, 0.7)


if __name__ == "__main__":
    main()
