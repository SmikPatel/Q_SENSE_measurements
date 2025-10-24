import numpy as np
from utils_m2_factorize import factorize_state, get_indices_mapping_2_wvn, expand_tensor_product, partial_trace_einsum
import sys
import pickle
from utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from utils_states import compress_state, convert_dense_format_to_sparse_format, convert_TZ_format_to_sparse_format

def test_with_real_data(bond_length):
    """Test with actual data from pickle file."""
    filename = f'h2o_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'

    try:
        with open(filename, 'rb') as f:
            (
                list_list_refCSF,
                list_list_Uext_mp2_CSF,
                list_list_Uext_mp2_ampld,
                list_list_Uext_opt_ampld,
                list_orb_rot,
                x_orbrot,
                Enuc,
                obt_spatial,
                tbt_spatial
            ) = pickle.load(f)

        if len(list_orb_rot) != 0:
            obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
        else:
            obt = obt_phys_spatial_to_spin(obt_spatial)
            tbt = tbt_phys_spatial_to_spin(tbt_spatial)

        # Get the first basis state
        Norb = len(obt_spatial)

        Nqubits = obt.shape[0]
        Norb    = Nqubits // 2
        dim     = 2 ** Nqubits

        tz_states       = []
        reference_states = []
        W_amplitude_mapping = []
  
        for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
            for j, ucsf in enumerate(ucsf_list):
                tz_states.append(ucsf)
                reference_states.append(list_list_refCSF[i][j])
                W_amplitude_mapping.append(i)

        print(f"\nTesting with real data (bond length {bond_length})")
        print("=" * 60)

        state_index = 36

        statevectors = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in tz_states]
        tapered_state = convert_dense_format_to_sparse_format(compress_state(statevectors[state_index].toarray()[0]))

        # Get index mapping
        index_mapping, state_type = get_indices_mapping_2_wvn(reference_states[state_index], list_list_Uext_mp2_ampld[W_amplitude_mapping[state_index]], Norb)
        print(f"Index mapping for state {state_index}: {index_mapping}")
        print(f"State type: {state_type}")
        # Extract S_w, S_v, S_n from mapping
        S_w = [idx for idx, label in index_mapping.items() if label == 'W']
        S_v = [idx for idx, label in index_mapping.items() if label == 'V']
        S_n = [idx for idx, label in index_mapping.items() if label == 'N']

        print(f"S_w (spatial): {S_w}")
        print(f"S_v (spatial): {S_v}")
        print(f"S_n (spatial): {S_n}")
        print()

        print(f"Tapered state: {tapered_state}")

        full_state = np.array(tapered_state.todense()).flatten()
        traced_out_state, _ = partial_trace_einsum(full_state, S_w, Norb)
        print(f"State after tracing out S_w (norm = {np.linalg.norm(traced_out_state):.6f}):")
        print(traced_out_state)
        print()

        # Factorize
        factorization_dict = factorize_state(tapered_state, S_w, S_v, S_n, state_type)


        print("Factorization result:")
        print("-" * 60)
        for key, value in factorization_dict.items():
            print(f"Qubits {key}: state vector length {len(value)}, norm = {np.linalg.norm(value):.6f}")
            print(value)

        psi = expand_tensor_product(factorization_dict, Norb)

        # Check if reconstructed state matches original (up to global phase)
        print("Verification:")
        print("-" * 60)

        # Check both positive and negative phase
        diff_positive = np.linalg.norm(psi - full_state)
        diff_negative = np.linalg.norm(psi + full_state)
        difference = min(diff_positive, diff_negative)

        print(f"||psi - full_state|| = {diff_positive:.10e}")
        print(f"||psi + full_state|| = {diff_negative:.10e}")
        print(f"Minimum difference = {difference:.10e}")

        if difference < 1e-3:
            print("✓ Reconstruction successful: states match (up to global phase)!")
        else:
            print("✗ Reconstruction failed: states do not match")
            print(f"Relative error: {difference / np.linalg.norm(full_state):.10e}")

    except FileNotFoundError:
        print(f"Data file not found: {filename}")
        print("Skipping real data test")


if __name__ == "__main__":

    # Test with real data if available
    if len(sys.argv) > 1:
        bond_length = sys.argv[1]
        test_with_real_data(bond_length)
    else:
        print("\nTo test with real data, provide bond_length as argument:")
        print("  python test_factorize.py <bond_length>")
