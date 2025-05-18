# In this file, we will calculate the projector matrix for a given set of rdm and fragments.

import numpy as np
import sparse

# def get_projector_matrix(rdm: np.ndarray, fragment: np.ndarray, reorder_idx:np.ndarray , threshold: float = 1e-5) -> np.ndarray:
#      """
#      Calculate the DMET projector matrix for a given one-particle RDM and fragment indices.

#      Parameters
#      ----------
#      rdm : (M, M) ndarray
#           The global one-particle reduced density matrix in AO basis.
#      fragment : (n_f,) ndarray of ints
#           The indices of the fragment orbitals (0-based).
#      threshold : float
#           Singular value threshold to select bath orbitals.

#      Returns
#      -------
#      P : (M, M) ndarray
#           The DMET embedding projector onto the fragment + bath space.
#      """
#      # print(reorder_idx)
#      rdm = rdm[np.ix_(reorder_idx, reorder_idx)]
#      shape = rdm.shape
#      length = len(fragment)
#      P_identity = np.eye(shape[0],length)
#      rdm_bath = rdm[length:, length:]
#      eigvals, eigvecs = np.linalg.eigh(rdm_bath)
#      idx = eigvals.argsort()[::-1]
#      eigvals = eigvals[idx]
#      eigvecs = eigvecs[:, idx]
#      number_of_bath_orbitals = np.sum((eigvals > threshold) & (eigvals < 1 - threshold))
#      eigvals = eigvals[:number_of_bath_orbitals]
#      eigvecs = eigvecs[:, :number_of_bath_orbitals]
#      Pbath = np.zeros((shape[0],number_of_bath_orbitals), dtype=complex)
#      Pbath[length:, :] = eigvecs
#      Pbath[:length, :] = np.zeros((length,number_of_bath_orbitals), dtype=complex)
#      # find the projector matrix [P_identity, P_bath]
#      # print(f"identity projector matrix: {P_identity}")
#      # print(f"Bath projector matrix: {Pbath}")
#      P=np.concatenate((P_identity, Pbath), axis=1)
     
#      # check if the projector matrix is unitary
#      # Pdagger = np.conjugate(P.T)
     
#      # PdaggerP = np.dot(Pdagger, P)
#      # if np.allclose(PdaggerP, np.eye(PdaggerP.shape[0]), atol=1e-5):
#      #      print("The projector matrix is unitary.")
#      # else:
#      #      print("The projector matrix is not unitary.")
#      #      print(f"PdaggerP: {PdaggerP}")
#      #      print(f"identity: {np.eye(PdaggerP.shape[0])}")
#      #      raise ValueError("The projector matrix is not unitary.")
#      return P
     


def get_projector_reorder_idxs(fragments: list, rdm_length : int) -> list:
    """
     Get the reorder indices for the projector matrix.
     Parameters
     ----------
     fragments : list of (n_f,) ndarray of ints
         The indices of the fragment orbitals (0-based).
     rdm_length : int
         The length of the RDM.
     Returns
     -------
     reorder_idxs : list of (M,) ndarray of ints
         The reorder indices for the projector matrix.
    """
    reorder_idxs = []
    for frag in fragments:
        rest  = np.setdiff1d(np.arange(rdm_length), frag)
        reorder_idxs.append(np.concatenate((frag, rest)))
    return reorder_idxs


import numpy as np


def get_projector_matrix(rdm: np.ndarray,
                         fragment: np.ndarray,
                         reorder_idx: np.ndarray,
                         threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate the DMET embedding projector matrix for a given one-particle RDM and fragment indices.

    Parameters
    ----------
    rdm : (M, M) ndarray
        The global one-particle reduced density matrix in AO basis.
    fragment : (f,) ndarray of ints
        The indices of the fragment orbitals (0-based).
    reorder_idx : (M,) ndarray of ints
        A permutation array to reorder the orbitals such that fragment indices come first.
    threshold : float
        Threshold to select bath orbitals (0 < occ < 1).

    Returns
    -------
    P : (M, f + b) ndarray
        The unitary projector onto the fragment plus bath space, where b = number of bath orbitals.
    """
    # 1. Reorder the RDM so that fragment orbitals are first
    rdm_full = rdm[np.ix_(reorder_idx, reorder_idx)]
    M = rdm_full.shape[0]
    f = len(fragment)

    # 2. Partition the RDM
    gamma_ff = rdm_full[:f, :f]
    gamma_fe = rdm_full[:f, f:]
    gamma_ef = rdm_full[f:, :f]

    # 3. Solve eigenproblem for coupling
    #    C = gamma_ef (shape (M-f, f))
    C = gamma_ef
    CCt = C @ C.T
    occvals, U = np.linalg.eigh(CCt)
    # sort descending
    idx = np.argsort(occvals)[::-1]
    occvals = occvals[idx]
    U = U[:, idx]

    # 4. Select bath orbitals by occupation threshold
    mask = (occvals > threshold) & (occvals < 1 - threshold)
    occvals = occvals[mask]
    U = U[:, mask]
    b = U.shape[1]

    # 5. Construct bath orbitals on full space: b_i = (1/sqrt(lambda_i)) * gamma_fe @ u_i
    baths = []
    for i in range(b):
        lam = occvals[i]
        u = U[:, i]
        # environment->fragment coupling gives bath amplitudes on fragment space
        coeff_f = (gamma_fe @ u) / np.sqrt(lam)
        # build full-space vector
        bi = np.zeros(M, dtype=complex)
        bi[:f] = coeff_f
        bi[f:] = u
        baths.append(bi)
    B = np.column_stack(baths) if b > 0 else np.zeros((M, 0), dtype=complex)

    # 6. Stack fragment identity and bath orbitals
    P0 = np.hstack([np.eye(M, f, dtype=complex), B])

    # 7. Orthonormalize via QR
    Q, _ = np.linalg.qr(P0)
    return Q
