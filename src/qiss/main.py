from .optimize import CMA
from .loadelems import Elem


def main():

    opt = CMA(target_loss=1e-6, max_iterations=100)
    ca = Elem("Ca")


if __name__ == "__main__":
    args = vars()
    main(**args)
