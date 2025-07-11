import cudaq
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import FermionOperator
from itertools import combinations
from DMET.ProblemSolver import ProblemSolver
from scipy.optimize import minimize

class EigenSolver(ProblemSolver):
    def __init__(self):
        super().__init__()

class EigenSolver(ProblemSolver):
    def __init__(self):
        super().__init__()

    def solve(self, hamiltonian: FermionOperator, number_of_orbitals: int,
              number_of_electrons: int, **kwargs):
        if not isinstance(hamiltonian, FermionOperator):
            raise TypeError("Hamiltonian must be a FermionOperator")
        cudaq_ham = cudaq.SpinOperator(self.ensure_real_coefficients(jordan_wigner(hamiltonian)))
        # Step 2: Define particle-number-conserving ASWAP ansatz
        def make_ansatz(n_qubits, number_of_electrons=None):
            assert number_of_electrons is not None, "number_of_electrons must be provided"
            assert number_of_electrons <= n_qubits, "number_of_electrons must be less than or equal to n_qubits"
            assert number_of_electrons >= 0, "number_of_electrons must be non-negative"
            @cudaq.kernel
            def kernel(params: list[float]):
                # conservation of particle number
                qubits = cudaq.qvector(n_qubits)
                for i in range(number_of_electrons):
                    x(qubits[i])

                cudaq.kernels.uccsd(qubits, params, number_of_electrons, n_qubits)
                cudaq.kernels.uccsd(qubits, params, number_of_electrons, n_qubits)
                cudaq.kernels.uccsd(qubits, params, number_of_electrons, n_qubits)
                    
            return kernel,3* cudaq.kernels.uccsd_num_parameters( number_of_electrons, n_qubits)
        
        kernel, params = make_ansatz(number_of_orbitals, number_of_electrons)
        
        def cost_function(opt_params):
            energy = cudaq.observe(kernel, cudaq_ham, opt_params).expectation()
            return energy.real
        
        # Step 1: Optimize the ansatz parameters
        initial_params = [np.random.random() for i in range(params)]  # Random initial parameters
        result = minimize(cost_function, initial_params, method='COBYLA')
        opt_params = result.x
        energy = result.fun
        # Step 4: Compute 1-RDM
        one_rdm, two_rdm = self.get_rdm(kernel, opt_params, number_of_orbitals)
        return energy, one_rdm, two_rdm

    def ensure_real_coefficients(self,qubit_ham):
        for term in qubit_ham.terms:
        #    print(f"Term: {term}, Coefficient: {qubit_ham.terms[term]}")
            if qubit_ham.terms[term].imag > 1e-10:
                raise ValueError("Imaginary coefficients found in the Hamiltonian, which is not allowed.")
            qubit_ham.terms[term] = qubit_ham.terms[term].real  # Ensure coefficients are real
        return qubit_ham
    
    def get_rdm(self, kernel, opt_params, number_of_orbitals):
        import numpy as np
        from openfermion import FermionOperator
        from openfermion.transforms import jordan_wigner
        import cudaq

        one_rdm = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                # Hermitian symmetrization
                op = FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}")
                spin_op = cudaq.SpinOperator(self.ensure_real_coefficients(jordan_wigner(op)))
                val = cudaq.observe(kernel, spin_op, opt_params).expectation()
                one_rdm[p, q] = val/2

        two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                            number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                for r in range(number_of_orbitals):
                    for s in range(number_of_orbitals):
                        # Hermitian symmetrization of two-body operator
                        op1 = FermionOperator(f"{p}^ {q}^ {s} {r}")
                        op2 = FermionOperator(f"{r}^ {s}^ {q} {p}")
                        op = op1 + op2
                        spin_op = cudaq.SpinOperator(self.ensure_real_coefficients(jordan_wigner(op)))
                        val = cudaq.observe(kernel, spin_op, opt_params).expectation()
                        two_rdm[p, r, q, s] = val/2

        return one_rdm, two_rdm



if __name__ == "__main__":
    # Example usage
    import openfermion
    import openfermionpyscf
    from openfermion.transforms import jordan_wigner, get_fermion_operator
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import numpy as np
    import pickle
    eigen_solver = EigenSolver()
    # Test with a 4-orbital 2-body operator (adjust as needed)
    # H = FermionOperator("0^ 1", 1.0) + FermionOperator("1^ 0", 1.0) + FermionOperator("0^ 1 2 3^", -.5) + FermionOperator("1^ 0 3 2^", -.5)
    geometry = [('H', (0, 0, 0.0)),
                ('H', (0, 0, 0.74))]
    basis = 'sto3g'
    multiplicity = 1
    charge = 0
    molecule = openfermionpyscf.run_pyscf(
        openfermion.MolecularData(geometry, basis, multiplicity, charge))
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    H = get_fermion_operator(molecular_hamiltonian)
    solver = EigenSolver()
    print('aaaa')
    energy, rdm1, rdm2 = solver.solve(H, number_of_orbitals=4, number_of_electrons=2)
    print("Energy:", energy)
    print("1-RDM:\n", rdm1)
    print("2-RDM shape:", rdm2)
    with open("array.pkl", "wb") as f:
        pickle.dump(rdm2, f)