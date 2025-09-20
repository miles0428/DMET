import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import FermionOperator
from itertools import combinations
from DMET.ProblemSolver import ProblemSolver
from scipy.optimize import minimize
import time 
import concurrent.futures
import threading
from scipy.sparse import csc_array

try:
    import cudaq
    import cudaq_solvers as solvers
except ImportError:
    pass

# --- Standalone Functions ---
def ensure_real_coefficients(qubit_ham):
    for term in qubit_ham.terms:
        if qubit_ham.terms[term].imag > 1e-10:
            raise ValueError("Imaginary coefficients found in the Hamiltonian, which is not allowed.")
        qubit_ham.terms[term] = qubit_ham.terms[term].real
    return qubit_ham

def make_ansatz(n_qubits, number_of_electrons=None, depth=1, mode='cudaq-vqe'):
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
            for k in range(n_qubits):
                for i in range(n_qubits):
                    if i > k:
                        cx(qubits[(k) % n_qubits], qubits[i])
                        rz(params[param_idx + 0], qubits[(k) % n_qubits])
                        rz(np.pi, qubits[(k) % n_qubits])
                        ry(params[param_idx + 1], qubits[(k) % n_qubits])
                        ry(np.pi / 2, qubits[(k) % n_qubits])
                        cx(qubits[i], qubits[(k) % n_qubits])
                        ry(-np.pi / 2, qubits[(k) % n_qubits])
                        ry(-params[param_idx + 1], qubits[(k) % n_qubits])
                        rz(-np.pi, qubits[(k) % n_qubits])
                        rz(-params[param_idx + 0], qubits[(k) % n_qubits])
                        cx(qubits[(k) % n_qubits], qubits[(i) % n_qubits])
                        cz(qubits[(k) % n_qubits], qubits[(i) % n_qubits])
                        param_idx += 2
    num_params = n_qubits * (n_qubits - 1) * depth

    @cudaq.kernel
    def kernel_no_params():
        qubits = cudaq.qvector(n_qubits)
        for i in range(number_of_electrons):
            x(qubits[i])

    if mode in ['classical', 'cudaq-vqe', 'cudaqx-vqe']:
        kernel_r = kernel
    else:
        kernel_r = kernel_no_params
    return kernel_r, num_params

def get_rdm(kernel, opt_params, number_of_orbitals, simulate_options, i=0, num_qpus=1, N=None):
    if simulate_options.get("async_observe", False):
        if simulate_options.get("hybridtest", False):
            cudaq.set_target('nvidia', option='mqpu')
    one_rdm = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
    vals = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.dtype(object))
    for p in range(number_of_orbitals):
        for q in range(number_of_orbitals):
            op = FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}")
            spin_op = cudaq.SpinOperator(ensure_real_coefficients(jordan_wigner(op)))
            if not simulate_options.get("async_observe", False):
                vals[p][q] = cudaq.observe(kernel, spin_op, opt_params).expectation()
            else:
                vals[p][q] = cudaq.observe_async(kernel, spin_op, opt_params, qpu_id=i % num_qpus)
                i += 1
    time.sleep(0)
    for p in range(number_of_orbitals):
        for q in range(number_of_orbitals):
            if not simulate_options.get("async_observe", False):
                val = vals[p][q]
            else:
                val = vals[p][q].get().expectation()
            one_rdm[p, q] = val / 2
    two_rdm = np.zeros((number_of_orbitals, number_of_orbitals, number_of_orbitals, number_of_orbitals), dtype=np.complex128)
    vals = np.zeros((number_of_orbitals, number_of_orbitals, number_of_orbitals, number_of_orbitals), dtype=np.dtype(object))
    for p in range(number_of_orbitals):
        for q in range(number_of_orbitals):
            for r in range(number_of_orbitals):
                for s in range(number_of_orbitals):
                    op1 = FermionOperator(f"{p}^ {q}^ {s} {r}")
                    op2 = FermionOperator(f"{r}^ {s}^ {q} {p}")
                    op = op1 + op2
                    spin_op = cudaq.SpinOperator(ensure_real_coefficients(jordan_wigner(op)))
                    if not simulate_options.get("async_observe", False):
                        vals[p][q][r][s] = cudaq.observe(kernel, spin_op, opt_params).expectation()
                    else:
                        vals[p][q][r][s] = cudaq.observe_async(kernel, spin_op, opt_params, qpu_id=i % num_qpus)
                        i += 1
    for p in range(number_of_orbitals):
        for q in range(number_of_orbitals):
            for r in range(number_of_orbitals):
                for s in range(number_of_orbitals):
                    if not simulate_options.get("async_observe", False):
                        val = vals[p][q][r][s]
                    else:
                        val = vals[p][q][r][s].get().expectation()
                    two_rdm[p, r, q, s] = val / 2
    if simulate_options.get("hybridtest", False):
        import os
        if N is not None:
            os.environ["CUDAQ_MGPU_NQUBITS_THRESH"] = str(N - 2)
            cudaq.set_target('nvidia', option='mgpu')
    return one_rdm, two_rdm

def _solve(hamiltonian, number_of_orbitals, number_of_electrons, depth=2, simulate_options=None, i=0, num_qpus=1):
    if simulate_options is None:
        simulate_options = {}
    if not isinstance(hamiltonian, FermionOperator):
        raise TypeError("Hamiltonian must be a FermionOperator")
    if not (num_qpus > 1):
        simulate_options["async_observe"] = False
    N = number_of_orbitals
    if simulate_options.get("hybridtest", False):
        import os
        os.environ["CUDAQ_MGPU_NQUBITS_THRESH"] = str(N - 2)
        cudaq.set_target('nvidia', option='mgpu')
    cudaq_ham = cudaq.SpinOperator(ensure_real_coefficients(jordan_wigner(hamiltonian)))
    kernel, params = make_ansatz(number_of_orbitals, number_of_electrons, depth=depth, mode=simulate_options.get("mode", "classical"))
    def cost_function(opt_params):
        if not simulate_options.get("async_observe", False):
            energy = cudaq.observe(kernel, cudaq_ham, opt_params).expectation()
        else:
            nonlocal i
            energy = cudaq.observe_async(kernel, cudaq_ham, opt_params, qpu_id=i % num_qpus)
            i += 1
            energy = energy.get().expectation()
        return energy.real
    initial_params = [np.random.random() for _ in range(params)]
    if simulate_options.get("mode", "classical") == "classical":
        result = minimize(cost_function, initial_params, method='COBYLA', options={'rhobeg': 0.5, 'maxiter': 500})
        opt_params = result.x
        energy = 0
    elif simulate_options.get("mode") == "cudaq-vqe":
        optimizer = cudaq.optimizers.COBYLA()
        energy, opt_params = cudaq.vqe(kernel, cudaq_ham, optimizer, len(initial_params))
    elif simulate_options.get("mode") == "cudaqx-vqe":
        initialX = [np.random.random() for _ in range(params)]
        energy, opt_params, all_data = solvers.vqe(kernel, cudaq_ham, initialX, optimizer=minimize, method='COBYLA')
    one_rdm, two_rdm = get_rdm(kernel, opt_params, number_of_orbitals, simulate_options, i=i, num_qpus=num_qpus, N=N)
    return energy, one_rdm, two_rdm

def _cudaq_preheat():
    # 預熱 CUDA context，減少第一次 kernel 啟動延遲
    try:
        @cudaq.kernel
        def warmup():
            q = cudaq.qvector(1)
            pass
        cudaq.observe(warmup, cudaq.SpinOperator('Z0'), [0.0])
    except Exception:
        pass

# --- Class Wrapper ---
class EigenSolver(ProblemSolver):
    def __init__(self, depth=2, **simulate_options):
        super().__init__()
        self.depth = depth
        try:
            self.num_qpus = cudaq.get_target().num_qpus()
        except Exception:
            self.num_qpus = 1
        self.simulate_options = simulate_options
        self.pools = [concurrent.futures.ProcessPoolExecutor(max_workers=1),
                      concurrent.futures.ProcessPoolExecutor(max_workers=1)]
        self.next_pool_idx = 0
        self._background_rebuild_thread = None
        # 預熱兩個 pool
        for pool in self.pools:
            pool.submit(_cudaq_preheat)

    def _background_rebuild(self, idx):
        # 關閉舊 pool，重建新 pool 並預熱
        try:
            self.pools[idx].shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        self.pools[idx] = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        self.pools[idx].submit(_cudaq_preheat)
        
    def _transform_hamiltonian_to_matrix(self, hamiltonian, number_of_orbitals, number_of_electrons=None) -> csc_array:
        """
        Convert a FermionOperator Hamiltonian into sparse matrix form.

        Args:
            hamiltonian (FermionOperator): The fermionic Hamiltonian.
            number_of_orbitals (int): Total number of spin-orbitals.

        Returns:
            csc_array: Sparse matrix representation of the Hamiltonian.

        Main Concept:
            Transforms the second-quantized Hamiltonian into a sparse matrix representation for numerical calculations.
        """
        from openfermion.linalg import get_ground_state, get_sparse_operator, get_number_preserving_sparse_operator
        if number_of_electrons is not None:
            return get_number_preserving_sparse_operator(hamiltonian, number_of_orbitals, number_of_electrons)
        else:
            return get_sparse_operator(hamiltonian, n_qubits=number_of_orbitals)

    def solve(self, hamiltonian, number_of_orbitals, number_of_electrons, **kwargs):
        
        if number_of_electrons == 0:
            # Vacuum state
            energy = 0  # extract constant part
            one_rdm = np.zeros((number_of_orbitals, number_of_orbitals))
            two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                                number_of_orbitals, number_of_orbitals))
            return energy, one_rdm, two_rdm
        
        elif number_of_electrons == number_of_orbitals:  
            hamiltonian_sparse = self._transform_hamiltonian_to_matrix(hamiltonian, number_of_orbitals, number_of_electrons)
            energy = hamiltonian_sparse[0, 0].real 
            one_rdm = np.eye(number_of_orbitals)
            two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                                number_of_orbitals, number_of_orbitals))
            # fill two_rdm properly if needed, e.g., fully occupied
            for p in range(number_of_orbitals):
                for q in range(number_of_orbitals):
                    for r in range(number_of_orbitals):
                        for s in range(number_of_orbitals):
                            if p == q and r == s:
                                two_rdm[p, q, r, s] = 1.0 
            # print(two_rdm)  
            return energy, one_rdm, two_rdm
        
        pool_idx = self.next_pool_idx
        future = self.pools[pool_idx].submit(
            _solve,
            hamiltonian,
            number_of_orbitals,
            number_of_electrons,
            self.depth,
            self.simulate_options,
            0,
            self.num_qpus
        )
        # 背景重建另一個 pool
        rebuild_idx = 1 - pool_idx

        self._background_rebuild_thread = threading.Thread(target=self._background_rebuild, args=(rebuild_idx,))
        self._background_rebuild_thread.daemon = True
        self._background_rebuild_thread.start()
        self.next_pool_idx = rebuild_idx
        self._background_rebuild_thread.join()
        return future.result()

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
