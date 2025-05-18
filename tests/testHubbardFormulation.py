from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
import numpy as np

if __name__ == "__main__":
    # Example usage
    one_body_hubbard = OneBodyHubbardFormulation(L=4, t=1.0, number_of_electrons=2)

    print(one_body_hubbard.H)
    density_matrix = one_body_hubbard.get_density_matrix(number_of_electrons=2)
