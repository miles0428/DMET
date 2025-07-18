import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import FermionOperator
from itertools import combinations
from DMET.ProblemSolver import ProblemSolver
from scipy.optimize import minimize

class EigenSolver(ProblemSolver):
    def with_default_kwargs(defaults):
        def decorator(func):
            def wrapper(*args,**kwargs):
                for key, value in defaults.items():
                    kwargs.setdefault(key, value)
                return func(*args,**kwargs)
            return wrapper
        return decorator
    @with_default_kwargs({'async_observe' : False})
    def __init__(self, depth = 2, **simulate_options):
        super().__init__()
        import cudaq
        self.i = 0
        self.depth = depth
        self.num_qpus = cudaq.get_target().num_qpus()
        self.simulate_options = simulate_options
        if not(self.num_qpus > 1):
            self.simulate_options["async_observe"] = False

    def solve(self, hamiltonian: FermionOperator, number_of_orbitals: int,
              number_of_electrons: int, **kwargs):
        if not isinstance(hamiltonian, FermionOperator):
            raise TypeError("Hamiltonian must be a FermionOperator")
        import cudaq
        cudaq_ham = cudaq.SpinOperator(self.ensure_real_coefficients(jordan_wigner(hamiltonian)))
        # Step 2: Define particle-number-conserving ASWAP ansatz
        def make_ansatz(n_qubits, number_of_electrons=None, depth = 1):
            assert number_of_electrons is not None, "number_of_electrons must be provided"
            assert number_of_electrons <= n_qubits, "number_of_electrons must be less than or equal to n_qubits"
            assert number_of_electrons >= 0, "number_of_electrons must be non-negative"
            
            @cudaq.kernel
            def kernel(params: list[float]):
                qubits = cudaq.qvector(n_qubits)
                for i in range(number_of_electrons):
                    x(qubits[i])
                param_idx = 0
                for j in range(depth):
                    for i in range(n_qubits):
                        cx(qubits[(i+1) % n_qubits], qubits[i])
                        rz(params[param_idx + 0], qubits[(i+1) % n_qubits])
                        rz(np.pi, qubits[(i+1) % n_qubits])
                        ry(params[param_idx + 1], qubits[(i+1) % n_qubits])
                        ry(np.pi / 2, qubits[(i+1) % n_qubits])
                        cx(qubits[i], qubits[(i+1) % n_qubits])
                        ry(-np.pi / 2, qubits[(i+1) % n_qubits])
                        ry(-params[param_idx + 1], qubits[(i+1) % n_qubits])
                        rz(-np.pi, qubits[(i+1) % n_qubits])
                        rz(-params[param_idx + 0], qubits[(i+1) % n_qubits])
                        cx(qubits[(i+1) % n_qubits], qubits[(i) % n_qubits])
                        param_idx += 2
            num_params = 2 * n_qubits * depth
            return kernel, num_params

        kernel, params = make_ansatz(number_of_orbitals, number_of_electrons, depth = self.depth)

        def cost_function(opt_params):
            if self.simulate_options["async_observe"] == False:
                energy = cudaq.observe(kernel, cudaq_ham, opt_params).expectation()

            if self.simulate_options["async_observe"] == True:
                energy = cudaq.observe_async(kernel, cudaq_ham, opt_params, qpu_id = self.i % self.num_qpus).expectation()
                self.i += 1
                print("self.i", self.i)
                energy = energy.get()
            
            return energy.real

        # Step 1: Optimize the ansatz parameters
        initial_params = [np.random.random() for i in range(params)]  # Random initial parameters

        result = minimize(
            cost_function,
            initial_params,
            method='COBYLA',
            options={'rhobeg': 0.5, 'maxiter': 500}
        )

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

        vals = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.dtype(object))

        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                # Hermitian symmetrization
                op = FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}")
                spin_op = cudaq.SpinOperator(self.ensure_real_coefficients(jordan_wigner(op)))
                if self.simulate_options["async_observe"] == False:
                    vals[p][q] = cudaq.observe(kernel, spin_op, opt_params).expectation()
                if self.simulate_options["async_observe"] == True:
                    vals[p][q] =cudaq.observe_async(kernel, spin_op, opt_params, qpu_id = self.i % self.num_qpus).expectation()
                    self.i += 1

        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                if self.simulate_options["async_observe"] == False:
                    val = vals[p][q]
                if self.simulate_options["async_observe"] == True:
                    val = vals[p][q].get()
                one_rdm[p, q] = val/2

        two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                            number_of_orbitals, number_of_orbitals), dtype=np.complex128)

        vals = np.zeros((number_of_orbitals, number_of_orbitals,
                        number_of_orbitals, number_of_orbitals), dtype=np.dtype(object))

        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                for r in range(number_of_orbitals):
                    for s in range(number_of_orbitals):
                        # Hermitian symmetrization of two-body operator
                        op1 = FermionOperator(f"{p}^ {q}^ {s} {r}")
                        op2 = FermionOperator(f"{r}^ {s}^ {q} {p}")
                        op = op1 + op2
                        spin_op = cudaq.SpinOperator(self.ensure_real_coefficients(jordan_wigner(op)))
                        if self.simulate_options["async_observe"] == False:
                            vals[p][q][r][s] = cudaq.observe(kernel, spin_op, opt_params).expectation()
                        if self.simulate_options["async_observe"] == True:
                            vals[p][q][r][s] = cudaq.observe_async(kernel, spin_op, opt_params, qpu_id = self.i % self.num_qpus).expectation()
                            self.i += 1

        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                for r in range(number_of_orbitals):
                    for s in range(number_of_orbitals):
                        if self.simulate_options["async_observe"] == False:
                            val = vals[p][q][r][s]
                        if self.simulate_options["async_observe"] == True:
                            val = vals[p][q][r][s].get()
                        two_rdm[p, r, q, s] = val / 2

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
    energy, rdm1, rdm2 = solver.solve(H, number_of_orbitals=4, number_of_electrons=2)
    print("Energy:", energy)
    print("1-RDM:\n", rdm1)
    print("2-RDM shape:", rdm2)
    with open("array.pkl", "wb") as f:
        pickle.dump(rdm2, f)
