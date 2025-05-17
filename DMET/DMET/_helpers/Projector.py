# In this file, we will calculate the projector matrix for a given set of rdm and fragments.

import numpy as np
import sparse

def get_projector_matrix(rdm: np.ndarray, fragment: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate the DMET projector matrix for a given one-particle RDM and fragment indices.

    Parameters
    ----------
    rdm : (M, M) ndarray
        The global one-particle reduced density matrix in AO basis.
    fragment : (n_f,) ndarray of ints
        The indices of the fragment orbitals (0-based).
    threshold : float
        Singular value threshold to select bath orbitals.

    Returns
    -------
    P : (M, M) ndarray
        The DMET embedding projector onto the fragment + bath space.
    """
    # Total number of orbitals
    M = rdm.shape[0]

    # Ensure fragment is array of ints and sorted
    frag = np.array(fragment, dtype=int)
    frag = np.unique(frag)

    # Environment indices are those not in fragment
    env = np.setdiff1d(np.arange(M), frag)

    # 1. Fragment-Environment coupling block D_FE
    D_FE = rdm[np.ix_(frag, env)]  # shape (n_f, M-n_f)

    # 2. SVD on D_FE to obtain bath singular vectors
    #    we decompose D_FE = U * S * Vh
    U, S, Vh = np.linalg.svd(D_FE, full_matrices=False)

    # 3. Select significant bath components
    #    singular values represent coupling strength
    sig_idx = np.where(S > threshold )[0]

    # 4. Bath orbital coefficients in environment subspace
    #    columns of Vh^H are environment orbitals; pick those for sig_idx
    bath_env = Vh.conj().T[:, sig_idx]  # shape (M-n_f, n_bath)

    # 5. Build the AO-coefficient matrix for embedding orbitals
    #    a) fragment orbitals: identity on fragment rows
    n_f = len(frag)
    C_frag = np.zeros((M, n_f), dtype=complex)
    for i, idx in enumerate(frag):
        C_frag[idx, i] = 1.0

    #    b) bath orbitals: place bath_env rows at environment indices
    n_b = bath_env.shape[1]
    C_bath = np.zeros((M, n_b), dtype=complex)
    C_bath[np.ix_(env, np.arange(n_b))] = bath_env

    #    c) stack fragment + bath
    C_emb = np.hstack((C_frag, C_bath))  # shape (M, 2*n_f or M, n_f+n_b)

    # 6. Construct projector P = C_emb * C_emb^H
    P = C_emb @ C_emb.conj().T

    # Convert P to sparse matrix using sparse library
    #     P_sparse = sparse.COO(P)

    return P
