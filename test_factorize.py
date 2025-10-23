import numpy as np
from utils_m2_factorize import factorize_state, get_indices_mapping_2_wvn, expand_tensor_product
import sys
import pickle

def test_factorize_example():
    """Test the factorize_state function with a simple example."""

    # Example basis state from your description:
    # [determinants, indices, coefficients]
    determinants = [
        np.array([1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1.]),
        np.array([1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0.]),
        np.array([1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1.]),
        np.array([1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0.])
    ]
    indices = [15993, 15990, 15801, 15798]
    coefficients = np.array([0.5, -0.5, -0.5, 0.5])

    basis_state = [determinants, indices, coefficients]

    # Example S_w, S_v, S_n (spatial orbital indices)
    # You'll need to set these based on your actual use case
    S_w = []  # No W orbitals in this example
    S_v = [2, 3]  # Example V orbitals (spatial indices)
    S_n = [0, 1, 4, 5, 6]  # Example N orbitals (spatial indices)

    print("Testing factorize_state function")
    print("=" * 60)
    print(f"Basis state indices: {indices}")
    print(f"Coefficients: {coefficients}")
    print(f"S_w (spatial): {S_w}")
    print(f"S_v (spatial): {S_v}")
    print(f"S_n (spatial): {S_n}")
    print()

    # Factorize the state
    factorization_dict = factorize_state(basis_state, S_w, S_v, S_n)

    print("Factorization result:")
    print("-" * 60)
    for key, value in factorization_dict.items():
        print(f"Qubits {key}:")
        print(f"  State vector (length {len(value)}): {value}")
        print(f"  Norm: {np.linalg.norm(value):.6f}")
        print()

    # Verify that tensor product reconstructs the original state
    print("Verification:")
    print("-" * 60)
    n_spin_orbitals = len(determinants[0])

    # Reconstruct full state from factorization
    reconstructed_state = expand_tensor_product(factorization_dict, n_spin_orbitals)

    # Original sparse state
    original_state = np.zeros(2**n_spin_orbitals)
    for idx, coef in zip(indices, coefficients):
        original_state[idx] = coef

    # Normalize original state for comparison
    original_state_normalized = original_state / np.linalg.norm(original_state)

    # Check if they match
    diff = np.linalg.norm(reconstructed_state - original_state_normalized)
    print(f"Difference between original and reconstructed: {diff:.10f}")

    if diff < 1e-10:
        print("✓ Factorization successful!")
    else:
        print("✗ Factorization may have issues")
        print(f"Original state non-zero entries:")
        for idx in indices:
            print(f"  [{idx}] = {original_state_normalized[idx]:.6f}")
        print(f"Reconstructed state non-zero entries:")
        nonzero_recon = np.nonzero(reconstructed_state)[0]
        for idx in nonzero_recon:
            print(f"  [{idx}] = {reconstructed_state[idx]:.6f}")


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

        # Get the first basis state
        basis_state = list_list_refCSF[0][0]
        Norb = len(obt_spatial)

        print(f"\nTesting with real data (bond length {bond_length})")
        print("=" * 60)

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
        factorization_dict = factorize_state(basis_state, S_w, S_v, S_n)

        print("Factorization result:")
        print("-" * 60)
        for key, value in factorization_dict.items():
            print(f"Qubits {key}: state vector length {len(value)}, norm = {np.linalg.norm(value):.6f}")

    except FileNotFoundError:
        print(f"Data file not found: {filename}")
        print("Skipping real data test")


if __name__ == "__main__":
    # Test with simple example
    test_factorize_example()

    # Test with real data if available
    if len(sys.argv) > 1:
        bond_length = sys.argv[1]
        test_with_real_data(bond_length)
    else:
        print("\nTo test with real data, provide bond_length as argument:")
        print("  python test_factorize.py <bond_length>")
