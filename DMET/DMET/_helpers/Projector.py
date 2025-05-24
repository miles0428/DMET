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
     # print(reorder_idx)
     rdm = rdm[np.ix_(reorder_idx, reorder_idx)]
     shape = rdm.shape
     length = len(fragment)
     P_identity = np.eye(shape[0],length)
     rdm_bath = rdm[length:, length:]
     eigvals, eigvecs = np.linalg.eigh(rdm_bath)
     idx = eigvals.argsort()
     eigvals = eigvals[idx]
    #  print(f"eigvals: {eigvals}")
     eigvecs = eigvecs[:, idx]
     number_of_bath_orbitals = np.sum((eigvals > threshold) & (eigvals < 1 - threshold))
     bath_orbital_indices = np.where((eigvals > threshold) & (eigvals < 1 - threshold))[0]
    #  print(f"bath_orbital_indices: {bath_orbital_indices}")
     #  eigvals = eigvals[:number_of_bath_orbitals]
     #  eigvecs = eigvecs[:, :number_of_bath_orbitals]
    #  number_of_orbitals_close_to_zero = np.sum(eigvals < threshold)
    #  number_of_orbitals_close_to_one = np.sum(eigvals > 1 - threshold)
    #  eigvals = eigvals[number_of_orbitals_close_to_zero:shape[0]-number_of_orbitals_close_to_one]
    #  eigvecs = eigvecs[:, number_of_orbitals_close_to_zero:shape[0]-number_of_orbitals_close_to_one]
    #  print(f"number_of_bath_orbitals: {number_of_bath_orbitals}")
     eigvals = eigvals[bath_orbital_indices]
    #  print(f"eigvals: {eigvals}")
     eigvecs = eigvecs[:, bath_orbital_indices]
    #  print(f"eigvals: {eigvals}")
    #  print(f"eigvecs: {eigvecs}")
     Pbath = np.zeros((shape[0],number_of_bath_orbitals), dtype=complex)
     Pbath[length:, :] = eigvecs
     Pbath[:length, :] = np.zeros((length,number_of_bath_orbitals), dtype=complex)

     P=np.concatenate((P_identity, Pbath), axis=1)
     # check orthonormality
     assert np.allclose(P.T.conj() @ P, np.eye(P.shape[1]), atol=1e-10), "Projector matrix is not orthonormal"  
    
     return P
     


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

# def get_projector_matrix(rdm: np.ndarray,
#                          fragment: np.ndarray,
#                          reorder_idx: np.ndarray,
#                          threshold: float = 1e-5) -> np.ndarray:
#     """
#     Generate the projector matrix for a given fragment.

#     Args:
#         onebody_rdm (np.ndarray): The one-body reduced density matrix.
#         fragment (np.ndarray): The indices of the fragment.
#         reorder_idx (np.ndarray): The reordering indices for the orbitals.
#         bath_threshold (float): The threshold for bath orbitals.

#     Returns:
#         np.ndarray: The projector matrix for the fragment.

#     Main Concept:
#         Constructs a projector matrix to embed the fragment into the full system.

#     Math Detail:
#         The projector matrix is computed as:
#             P = \sum_{i \in fragment} |i><i|
#         where |i> are the basis states of the fragment.
#     """
#     # 1. Reorder the RDM so that fragment orbitals are first
#     rdm_full = rdm[np.ix_(reorder_idx, reorder_idx)]
#     M = rdm_full.shape[0]
#     f = len(fragment)

#     # 2. Partition the RDM
#     gamma_ff = rdm_full[:f, :f]
#     gamma_fe = rdm_full[:f, f:]
#     gamma_ef = rdm_full[f:, :f]

#     # 3. Solve eigenproblem for coupling
#     #    C = gamma_ef (shape (M-f, f))
#     C = gamma_ef
#     CCt = C @ C.T
#     occvals, U = np.linalg.eigh(CCt)
#     # sort descending
#     idx = np.argsort(occvals)[::-1]
#     occvals = occvals[idx]
#     U = U[:, idx]

#     # 4. Select bath orbitals by occupation threshold
#     mask = (occvals > threshold) & (occvals < 1 - threshold)
#     occvals = occvals[mask]
#     U = U[:, mask]
#     b = U.shape[1]

#     # 5. Construct bath orbitals on full space: b_i = (1/sqrt(lambda_i)) * gamma_fe @ u_i
#     baths = []
#     for i in range(b):
#         lam = occvals[i]
#         u = U[:, i]
#         # environment->fragment coupling gives bath amplitudes on fragment space
#         coeff_f = (gamma_fe @ u) / np.sqrt(lam)
#         # build full-space vector
#         bi = np.zeros(M, dtype=complex)
#         bi[:f] = coeff_f
#         bi[f:] = u
#         baths.append(bi)
#     B = np.column_stack(baths) if b > 0 else np.zeros((M, 0), dtype=complex)

#     # 6. Stack fragment identity and bath orbitals
#     P0 = np.hstack([np.eye(M, f, dtype=complex), B])

#     # 7. Orthonormalize via QR
#     Q, _ = np.linalg.qr(P0)
#     return Q
