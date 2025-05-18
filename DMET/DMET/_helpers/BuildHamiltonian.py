from openfermion import FermionOperator
import numpy as np

def build_hamiltonian_from_one_two_body(h1_emb: np.ndarray, h2_emb: np.ndarray, threshold: float = 1e-12) -> FermionOperator:
    """
    Construct a FermionOperator Hamiltonian from embedded one-body and two-body integrals.

    Args:
        h1_emb (np.ndarray): Embedded one-body integrals, shape (n_emb, n_emb)
        h2_emb (np.ndarray): Embedded two-body integrals, shape (n_emb, n_emb, n_emb, n_emb)
        threshold (float): Cutoff below which coefficients are ignored

    Returns:
        FermionOperator: The second-quantized Hamiltonian constructed from the inputs
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
    for p in range(n_emb):
        for q in range(n_emb):
            for r in range(n_emb):
                for s in range(n_emb):
                    coeff = h2_emb[p, q, r, s]
                    if abs(coeff) > threshold:
                        h_frag += FermionOperator(
                            ((p, 1), (q, 1), (s, 0), (r, 0)),
                            0.5 * coeff
                        )

    return h_frag
