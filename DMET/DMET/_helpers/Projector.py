# In this file, we will calculate the projector matrix for a given set of rdm and fragments.

import numpy as np

def get_projector_matrix(rdm: np.ndarray, fragment: np.ndarray, reorder_idx:np.ndarray , threshold: float = 1e-3) -> np.ndarray:
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
    number_of_occupations = np.trace(rdm)
    number_of_occupations = int(np.round(number_of_occupations))
    rdm = rdm[np.ix_(reorder_idx, reorder_idx)]
    shape = rdm.shape
    length = len(fragment)
    P_identity = np.eye(shape[0],length)
    rdm_bath = rdm[length:, length:]
    eigvals, eigvecs = np.linalg.eigh(rdm_bath)
    num_one = np.sum(np.isclose(eigvals, 1, atol=threshold))
    num_one = int(np.round(num_one))
    number_of_embedded_electrons = number_of_occupations - num_one
    idx = np.abs(eigvals-1).argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    number_of_bath_orbitals = np.sum((eigvals > threshold) & (eigvals < 1 - threshold))
    bath_orbital_indices = np.where((eigvals > threshold) & (eigvals < 1 - threshold))[0]
    eigvals = eigvals[bath_orbital_indices]
    eigvecs = eigvecs[:, bath_orbital_indices]
    Pbath = np.zeros((shape[0],number_of_bath_orbitals), dtype=complex)
    Pbath[length:, :] = eigvecs
    Pbath[:length, :] = np.zeros((length,number_of_bath_orbitals), dtype=complex)
    P=np.concatenate((P_identity, Pbath), axis=1)
    assert np.allclose(P.T.conj() @ P, np.eye(P.shape[1]), atol=1e-10), "Projector matrix is not orthonormal"  

    return P, number_of_embedded_electrons
     


def get_projector_reorder_idxs(fragments: list, rdm_length : int) -> list:
    """
     Generate reordering indices for the orbitals based on the fragments.

     Args:
         fragments (List[np.ndarray]): A list of fragments.
         num_orbitals (int): The total number of orbitals in the system.

     Returns:
         List[np.ndarray]: A list of reordering indices for each fragment.

     Main Concept:
         Reorders the orbitals to group fragment orbitals together.

     Math Detail:
         The reordering indices are computed as:
             reorder_idx = [fragment_indices, rest_indices]
         where fragment_indices are the indices of the fragment orbitals and rest_indices are the remaining indices.
    """
    reorder_idxs = []
    for frag in fragments:
        rest  = np.setdiff1d(np.arange(rdm_length), frag)
        reorder_idxs.append(np.concatenate((frag, rest)))
    return reorder_idxs

