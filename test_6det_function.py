import numpy as np
from csf_categorizer import find_double_excitation_indices_6det

# Example based on your data structure
# This appears to be 4 determinants, not 6
example_basis_states = [
    np.array([1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1.]),
    np.array([1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0.]),
    np.array([1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1.]),
    np.array([1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0.])
]

print("Testing with 4 determinants from your example:")
print(f"Number of determinants: {len(example_basis_states)}")

# Let's analyze what changes between determinants
det0 = example_basis_states[0]
print(f"\nReference determinant (det0):")
print(f"  {det0}")

all_changing = set()
for i, det in enumerate(example_basis_states[1:], 1):
    diff = det0 - det
    changing = [idx for idx, val in enumerate(diff) if abs(val) > 0.5]
    print(f"\nDeterminant {i}:")
    print(f"  {det}")
    print(f"  Changing positions: {changing}")
    all_changing.update(changing)

print(f"\nAll positions that change: {sorted(all_changing)}")

# Check occupation in det0
occupied_in_det0 = [idx for idx in sorted(all_changing) if det0[idx] > 0.5]
virtual_in_det0 = [idx for idx in sorted(all_changing) if det0[idx] < 0.5]

print(f"Occupied in det0 (should be i,j): {occupied_in_det0}")
print(f"Virtual in det0 (should be a,b): {virtual_in_det0}")

# Test the function (though it expects 6 determinants)
result = find_double_excitation_indices_6det(example_basis_states)
print(f"\nFunction result: i={result[0]}, j={result[1]}, a={result[2]}, b={result[3]}")

# Convert to spatial orbitals
if all(x is not None for x in result):
    i_spatial = result[0] // 2
    j_spatial = result[1] // 2
    a_spatial = result[2] // 2
    b_spatial = result[3] // 2
    print(f"Spatial orbitals: i={i_spatial}, j={j_spatial}, a={a_spatial}, b={b_spatial}")
