from openfermion import FermionOperator
import numpy as np

def build_hamiltonian_from_one_two_body(h1_emb: np.ndarray, h2_emb: np.ndarray, threshold: float = 1e-12) -> FermionOperator:
    """
    Construct a FermionOperator Hamiltonian from embedded one-body and two-body integrals.

    Args:
        h1_emb (np.ndarray): Embedded one-body integrals, shape (n_emb, n_emb).
        h2_emb (np.ndarray): Embedded two-body integrals, shape (n_emb, n_emb, n_emb, n_emb).
        threshold (float): Cutoff below which coefficients are ignored.

    Returns:
        FermionOperator: The second-quantized Hamiltonian constructed from the inputs.

    Main Concept:
        Constructs a Hamiltonian in the second-quantized form using the provided one-body and two-body integrals.

    Math Detail:
        The Hamiltonian is constructed as:
            H = \sum_{pq} h1_emb[p, q] c_p^\dagger c_q
                +   \sum_{pqrs} h2_emb[p, q, r, s] c_p^\dagger c_q^\dagger c_s c_r
        where c_p^\dagger and c_q are creation and annihilation operators.
    """
    n_emb = h1_emb.shape[0]
    h_frag = FermionOperator()

    # Add one-body terms
    for p in range(n_emb):
        for q in range(n_emb):
            coeff = h1_emb[p, q]
            if abs(coeff) > threshold:
                h_frag += FermionOperator(((p, 1), (q, 0)), coeff)

    # Add two-body terms
    n_emb = h2_emb.shape[0]
    for k in range(n_emb):
        for l in range(n_emb):
            for m in range(n_emb):
                for n in range(n_emb):
                    coeff = h2_emb[k, l, m, n]
                    if abs(coeff) > threshold:
                        h_frag += FermionOperator(
                            ((k, 1), (m, 1), (n, 0), (l, 0)),
                            coeff
                        )

    return h_frag
