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
        print('done')
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
                    
            return kernel, cudaq.kernels.uccsd_num_parameters( number_of_electrons, n_qubits)
        
        kernel, params = make_ansatz(number_of_orbitals, number_of_electrons)
        
        def cost_function(opt_params):
            energy = cudaq.observe(kernel, cudaq_ham, opt_params).expectation()
            return energy.real
        
        # Step 1: Optimize the ansatz parameters
        initial_params = np.random.rand(params) * 0.1  # Random initial parameters
        result = minimize(cost_function, initial_params, method='COBYLA')
        opt_params = result.x
        energy = result.fun
        print("Optimized parameters:", opt_params)
        print("Optimized energy:", energy)
        print('done')
        # Step 3: Run VQE
        optimizer = cudaq.optimizers.COBYLA()

        print('done')
        # Step 4: Compute 1-RDM
        one_rdm, two_rdm = self.get_rdm(kernel, opt_params, number_of_orbitals)
        return energy, one_rdm, two_rdm
    
    def ensure_real_coefficients(self,qubit_ham):
        for term in qubit_ham.terms:
            print(f"Term: {term}, Coefficient: {qubit_ham.terms[term]}")
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
                one_rdm[p, q] = val

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
                        two_rdm[p, q, r, s] = val

        return one_rdm, two_rdm



if __name__ == "__main__":
    # Example usage
    H = FermionOperator("0^ 1", 1.0) + FermionOperator("1^ 0", 1.0)
    solver = EigenSolver()
    print('aaaa')
    energy, rdm1, rdm2 = solver.solve(H, number_of_orbitals=4, number_of_electrons=2)
    print("Energy:", energy)
    print("1-RDM:\n", rdm1)
    print("2-RDM shape:", rdm2.shape)

