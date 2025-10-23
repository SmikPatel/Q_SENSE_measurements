import numpy as np
from utils_m2_factorize import factorize_state, get_indices_mapping_2_wvn, expand_tensor_product
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
        basis_state = list_list_refCSF[0][0]
        Norb = len(obt_spatial)

        Nqubits = obt.shape[0]
        Norb    = Nqubits // 2
        dim     = 2 ** Nqubits

        tz_states       = []
  
        for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
            for j, ucsf in enumerate(ucsf_list):
                print(f"Basis state: {list_list_refCSF[i][j]}")
                tz_states.append(ucsf)

        print(f"\nTesting with real data (bond length {bond_length})")
        print("=" * 60)

        statevectors = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in tz_states]
        tapered_state = convert_dense_format_to_sparse_format(compress_state(statevectors[0].toarray()[0]))

        # Get index mapping
        index_mapping = get_indices_mapping_2_wvn(basis_state, list_list_Uext_mp2_ampld[0], Norb)

        # Extract S_w, S_v, S_n from mapping
        S_w = [idx for idx, label in index_mapping.items() if label == 'W']
        S_v = [idx for idx, label in index_mapping.items() if label == 'V']
        S_n = [idx for idx, label in index_mapping.items() if label == 'N']

        print(f"S_w (spatial): {S_w}")
        print(f"S_v (spatial): {S_v}")
        print(f"S_n (spatial): {S_n}")
        print()

        # Factorize
        factorization_dict = factorize_state(tapered_state, S_w, S_v, S_n)

        print("Factorization result:")
        print("-" * 60)
        for key, value in factorization_dict.items():
            print(f"Qubits {key}: state vector length {len(value)}, norm = {np.linalg.norm(value):.6f}")

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
