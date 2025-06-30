from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
import numpy as np

def get_one_body_matrix(fermion_operator, n_orbitals):
    """Extract the one-body matrix from a FermionOperator"""
    h_matrix = np.zeros((n_orbitals, n_orbitals), dtype=complex)
    for term, coeff in fermion_operator.terms.items():
        if len(term) == 2:
            p, q = term[0], term[1]
            if p[1] == 1 and q[1] == 0:  # a†_p a_q
                h_matrix[p[0], q[0]] = coeff
    return h_matrix

if __name__ == "__main__":
    L = 4
    t = 1.0
    U = 4.0
    one_body_hubbard = OneBodyHubbardFormulation(L, t, number_of_electrons=6)
    many_body_hubbard = ManyBodyHubbardFormulation(L, t, U)
    print("One-body Hamiltonian:", one_body_hubbard.H)
    # print("Many-body Hamiltonian:", many_body_hubbard.H)
    # get the analytic solution
    rdm = one_body_hubbard.get_density_matrix()
    print("One-body reduced density matrix:", rdm)
    # get the analytic solution
    # use eigen_solver to get the energy of the one body problem
    from scipy.sparse.linalg import eigsh
    from openfermion import get_sparse_operator
    one_body_hubbard_hamiltonian = one_body_hubbard.H
    # change one_body_hubbard_hamiltonian to a sparse matrix
    one_body_matrix = get_one_body_matrix(one_body_hubbard_hamiltonian, 2*L)
    one_body_matrix= one_body_matrix.real.round(10)

    print("One-body Hamiltonian matrix:", one_body_matrix)
    # get the eigenvalues and eigenvectors
    values, vectors = eigsh(
        one_body_matrix, k=2*L-1, v0=None, which='SA', maxiter=1e7
    )
    
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    # build the slater determinant
    slater_determinant = np.zeros((2*L, 6), dtype=complex)
    print('shape of slater_determinant:', slater_determinant.shape)
    print('shape of vectors:', vectors.shape)
    
    for i in range(6):
        slater_determinant[:, i] = vectors[:, i]
    
    density_matrix = np.dot(slater_determinant, slater_determinant.conjugate().T).round(10).real
    print("Density matrix from eigen solver:", density_matrix)
    # compare the density matrix from the eigen solver and the one from the analytic solution
    if np.allclose(density_matrix, rdm):
        print(sum(rdm-density_matrix))
        print("The density matrices are the same.")
    else:
        print("The density matrices are different.")
        
        
