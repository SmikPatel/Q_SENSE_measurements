# CSF Categorizer and Quantum State Factorization Documentation

## Overview

This document explains the functionality of the CSF (Configuration State Function) categorization system and related quantum state factorization utilities used in quantum chemistry calculations.

---

## csf_categorizer.py

### Purpose
The `csf_categorizer.py` module analyzes and categorizes Configuration State Functions (CSFs) in quantum chemistry calculations. CSFs are linear combinations of Slater determinants that represent electronic states with proper spin and spatial symmetry.

### Main Functionality

#### State Categories
The module classifies CSF states into 4 distinct types:

1. **Type 1: |HF⟩** - Hartree-Fock Reference State
   - Single determinant with coefficient 1.0
   - Represents the ground state reference configuration
   - No excitations present

2. **Type 2: E_{ia}|HF⟩** - Single or Double Excitation (2 determinants)
   - Two determinants with specific coefficient patterns
   - Can represent either:
     - Single excitation: electron moved from occupied orbital i to virtual orbital a
     - Double excitation: two electrons moved simultaneously
   - Typical coefficients: ±1/√2 (≈ ±0.707)

3. **Type 3: E_{ia}E_{jb}|HF⟩** - Double Excitation (4 determinants)
   - Four determinants with coefficients ±0.5
   - Represents product of two single excitations
   - Involves two occupied orbitals (i, j) and two virtual orbitals (a, b)

4. **Type 4: Complex Multi-determinant States** (6+ determinants)
   - Six or more determinants with complex coefficient patterns
   - May involve √3-based coefficients (±1/√3 ≈ ±0.577, ±1/(2√3) ≈ ±0.289)
   - Represents more complex electronic configurations

### Key Functions

#### `categorize_csf_state(csf_state)`
**Purpose:** Main classification function that determines the type of a CSF state and extracts relevant quantum number indices.

**Input:**
- `csf_state`: CSF state in format `[basis_states, indices, coefficients]`
  - `basis_states`: List of determinants (occupation number vectors)
  - `indices`: State indices
  - `coefficients`: Linear combination coefficients

**Output:**
- Tuple: `(type_number, i, j, a, b)`
  - `type_number`: Integer 1-4 indicating state category (0 for unknown)
  - `i, j`: Occupied spin-orbital indices (None if not applicable)
  - `a, b`: Virtual spin-orbital indices (None if not applicable)

**Logic:**
1. Counts determinants in the state
2. Checks coefficient patterns
3. Extracts orbital indices using specialized helper functions
4. Returns classification and indices

#### `find_single_excitation_indices(det1, det2)`
**Purpose:** Identifies the occupied→virtual orbital transition in a single excitation.

**Algorithm:**
1. Computes difference between two determinants
2. Finds where det1 has 1 and det2 has 0 (occupied orbital i)
3. Finds where det1 has 0 and det2 has 1 (virtual orbital a)
4. Returns (i, a) if exactly one electron differs, otherwise (None, None)

#### `find_double_excitation_indices(det1, det2)`
**Purpose:** Identifies the two occupied→virtual orbital transitions in a double excitation.

**Algorithm:**
1. Computes difference between two determinants
2. Finds two occupied orbitals (i, j) that differ
3. Finds two virtual orbitals (a, b) that differ
4. Returns (i, j, a, b) if exactly two electrons differ

#### `find_double_excitation_indices_4det(basis_states)`
**Purpose:** Extracts double excitation indices from a 4-determinant state.

**Algorithm:**
1. Identifies spin-orbitals that vary across all 4 determinants
2. Groups spin-orbitals into spatial orbital pairs (2i, 2i+1)
3. Determines which spatial orbitals are occupied vs virtual in reference determinant
4. Returns spatial orbital indices (i, j, a, b)

#### `find_double_excitation_indices_6det(basis_states)`
**Purpose:** Extracts double excitation indices from a 6-determinant state.

**Algorithm:**
1. Compares all 6 determinants to find changing spin-orbitals
2. Converts spin-orbital indices to spatial orbital indices (divide by 2)
3. Identifies 4 changing spatial orbitals
4. Classifies as occupied (i, j) or virtual (a, b) based on reference determinant
5. Returns spatial orbital indices (i, j, a, b)

#### `analyze_all_csf_states(list_list_refCSF)`
**Purpose:** Batch analysis and reporting function for multiple CSF states.

**Features:**
- Categorizes all states in the input list
- Prints detailed analysis for each state
- Shows determinant count, coefficients, and orbital indices
- Provides summary statistics of state type distribution

---

## Key Functions from utils_m2_factorize.py

### 1. `factorize_state(basis_csr_state, S_w, S_v, S_n, state_type)`

**Purpose:** Factorizes a quantum state into tensor product components based on orbital classification into W (quantum-treated), V (rotation-affected), and N (invariant) blocks.

#### Input Parameters:
- `basis_csr_state`: Sparse CSR matrix representing the quantum state
  - Dimension: 2^n_spatial_orbitals
  - Stored in spatial orbital basis
- `S_w`: List of spatial orbital indices requiring quantum treatment (W block)
- `S_v`: List of spatial orbital indices affected by rotations (V block)
- `S_n`: List of spatial orbital indices that are invariant (N block)
- `state_type`: Integer (1-4) indicating CSF state category

#### Output:
Dictionary mapping spatial orbital tuples to normalized state vectors:
```python
{
    (orbitals_w...): psi_w,           # W block state vector
    (orbitals_v_1...): psi_v_1,       # V block state vector (part 1)
    (orbitals_v_2...): psi_v_2,       # V block state vector (part 2)
    (i,): psi_i,                      # Individual N block orbital states
    ...
}
```

#### Factorization Strategy:

1. **S_W Block:** All W orbitals kept as one entangled block
   - Orbitals requiring full quantum treatment
   - Cannot be factorized further

2. **S_V Block Treatment:**
   - **2 spatial orbitals:** Kept as one block
   - **4 spatial orbitals:**
     - **State Type 3:** Attempts to split into two pairs
       - Tries 3 possible pairings: (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)
       - Uses purity check to determine valid factorization
     - **State Type 4:** Kept as one 4-orbital block (entangled)
   - **Other cases:** Kept as single block

3. **S_N Block:** Each spatial orbital factorized separately
   - Individual orbitals treated independently
   - Creates (i,) entries for each orbital

#### Algorithm Steps:
1. Convert sparse CSR matrix to dense state vector
2. Determine number of spatial orbitals from state dimension
3. Extract W block using partial trace (if non-empty)
4. Handle V block based on size and state type:
   - Attempt multiple factorization strategies
   - Use purity checks to validate factorization
5. Factor each N block orbital individually
6. Return dictionary of factorized components

---

### 2. `partial_trace_einsum(psi, qubit_indices, n_total_qubits)`

**Purpose:** Computes the reduced density matrix by tracing out specified qubits, then extracts the dominant eigenstate. This operation is fundamental for obtaining marginal quantum states of subsystems.

#### Input Parameters:
- `psi`: Full state vector of dimension (2^n_total_qubits,)
- `qubit_indices`: List of qubit indices to keep (all others traced out)
- `n_total_qubits`: Total number of qubits in the system

#### Output:
Tuple: `(phi, is_pure)`
- `phi`: Normalized state vector of the subsystem (dominant eigenstate)
- `is_pure`: Boolean indicating if the reduced state is pure (purity ≈ 1)

#### Algorithm:

1. **Normalization:** Normalize input state vector

2. **Identify Trace-Out Qubits:**
   ```python
   trace_out = [i for i in range(n_total_qubits) if i not in qubit_indices]
   ```

3. **Einstein Summation Setup:**
   - Assign unique labels to each qubit index
   - Uses characters: 'abcd...xyzABC...XYZ' (max 26 qubits)
   - Create separate labels for ket and bra
   - For traced qubits: set ket_label = bra_label (contracts over this index)

4. **Build Einsum String:**
   ```
   Format: "ket_labels,bra_labels->keep_ket_labels+keep_bra_labels"
   Example: "abcd,aBcd->bBdD" (traces out qubits 0 and 2)
   ```

5. **Compute Reduced Density Matrix:**
   ```python
   psi_tensor = psi.reshape([2] * n_total_qubits)
   rho = np.einsum(einsum_str, psi_tensor, np.conjugate(psi_tensor))
   ```
   - Reshapes state into tensor form
   - Computes ρ = Tr_{traced}(|ψ⟩⟨ψ|)

6. **Extract Dominant Eigenstate:**
   - Compute eigendecomposition of ρ
   - Select eigenvector corresponding to largest eigenvalue
   - This is the "closest pure state" to the reduced density matrix

7. **Purity Check:**
   ```python
   purity = Tr(ρ²)
   is_pure = |purity - 1.0| < 1e-10
   ```
   - Purity = 1: Pure state (factorizable)
   - Purity < 1: Mixed state (entangled with traced qubits)

#### Physical Interpretation:
- **Pure result (is_pure=True):** The kept qubits are unentangled with the traced qubits; perfect factorization exists
- **Mixed result (is_pure=False):** Entanglement between kept and traced qubits; only approximate factorization possible
- Returns the "best" pure state approximation via dominant eigenvector

---

### 3. `get_indices_mapping_2_wvn(basis_state, mp2_amplitude, Norb)`

**Purpose:** Classifies spatial orbitals into three categories (W, V, N) based on their role in the quantum state and MP2 perturbation theory, then returns the state type.

#### Input Parameters:
- `basis_state`: List of occupied spin-orbital indices from CSF basis state
  - Format: `list_list_refCSF[i][j]`
- `mp2_amplitude`: List of MP2 (Møller-Plesset 2nd order) amplitude data
  - Format: `[[[i, a]], amplitude_value]`
  - Contains perturbative corrections to the wavefunction
- `Norb`: Total number of spatial orbitals in the system

#### Output:
Tuple: `(mapping_dict, state_type)`
- `mapping_dict`: Dictionary `{orbital_index: 'W' or 'V' or 'N'}`
- `state_type`: Integer (1-4) indicating CSF category

#### Orbital Classification:

1. **S_W (Quantum Treatment Block):**
   - Orbitals appearing in MP2 amplitude data
   - Require full quantum mechanical treatment
   - Cannot be approximated classically
   - Extracted from `mp2_amplitude[*][0][0]`

2. **S_V (Rotation-Affected Block):**
   - Orbitals involved in excitations from the CSF state
   - Determined by calling `categorize_csf_state(basis_state)`
   - Extracts indices k, l (occupied) and a, b (virtual)
   - Converts spin-orbitals to spatial orbitals (divide by 2)
   - Affected by unitary rotations but may not require full quantum treatment

3. **S_N (Invariant Block):**
   - All remaining orbitals
   - Invariant under both Vu and Wu rotations
   - Can be treated independently
   - Computed as: `Norb \ (S_W ∪ S_V)`

#### Algorithm Steps:

1. **Process MP2 Amplitudes:**
   ```python
   for amplitude_data in mp2_amplitude:
       indices = amplitude_data[0][0]  # [i, a]
       S_W.extend(indices)
   S_W = list(set(S_W))  # Remove duplicates
   ```

2. **Categorize CSF State:**
   ```python
   (state_type, k, l, a, b) = categorize_csf_state(basis_state)
   spatial_orbitals = [idx for idx in [k, l, a, b] if idx is not None]
   S_V = list(set(spatial_orbitals))
   ```

3. **Compute Invariant Set:**
   ```python
   S_N = [k for k in range(Norb) if k not in S_W and k not in S_V]
   ```

4. **Build Mapping Dictionary:**
   ```python
   mapping = {index: 'W' for index in S_W} |
             {index: 'V' for index in S_V} |
             {index: 'N' for index in S_N}
   ```

#### Physical Interpretation:

- **W orbitals:** Strong quantum correlations (MP2 corrections significant)
- **V orbitals:** Participate in electronic excitations (CSF structure)
- **N orbitals:** Spectator orbitals (weakly interacting)

This classification enables efficient hybrid quantum-classical algorithms by:
1. Treating W orbitals on quantum hardware (expensive but necessary)
2. Handling V orbitals with appropriate rotation strategies
3. Classically simulating N orbitals (cheap and accurate)

#### Usage in Workflow:
This function is typically called before `factorize_state()` to determine which orbitals belong to each block, enabling the state factorization strategy.

---

## Relationships Between Functions

```
get_indices_mapping_2_wvn()
    ↓ (provides S_W, S_V, S_N classification)
factorize_state()
    ↓ (uses partial_trace_einsum for each block)
partial_trace_einsum()
    ↓ (computes reduced states for each factor)
[Factorized state dictionary ready for quantum/classical processing]
```

The workflow:
1. Classify orbitals using `get_indices_mapping_2_wvn()`
2. Factorize the state into tensor products using `factorize_state()`
3. Each factorization uses `partial_trace_einsum()` to extract marginal states
4. Result: Dictionary of independent quantum states for efficient computation

---

## Mathematical Background

### Spin-Orbital vs Spatial Orbital Indexing
- **Spin-orbital index:** Labels both spatial position and spin (α or β)
  - Even indices (0, 2, 4, ...): α spin
  - Odd indices (1, 3, 5, ...): β spin
- **Spatial orbital index:** Labels only spatial position (spin-independent)
  - Conversion: `spatial_idx = spin_idx // 2`

### State Purity
For a density matrix ρ:
- **Purity:** P = Tr(ρ²)
- **Pure state:** P = 1 (can be written as |ψ⟩⟨ψ|)
- **Mixed state:** P < 1 (statistical mixture, entangled)

### Partial Trace
For composite system AB with state |ψ⟩_AB:
- **Reduced density matrix:** ρ_A = Tr_B(|ψ⟩⟨ψ|)
- **Interpretation:** State of subsystem A when B is ignored
- **Factorization:** |ψ⟩_AB = |φ⟩_A ⊗ |χ⟩_B only if ρ_A is pure

---

## Usage Example

```python
from csf_categorizer import categorize_csf_state
from utils_m2_factorize import (
    get_indices_mapping_2_wvn,
    factorize_state
)

# Step 1: Classify the CSF state
csf_state = [basis_states, indices, coefficients]
state_type, i, j, a, b = categorize_csf_state(csf_state)
print(f"State Type: {state_type}, Orbitals: i={i}, j={j}, a={a}, b={b}")

# Step 2: Map orbitals to W/V/N categories
mapping_dict, state_type = get_indices_mapping_2_wvn(
    basis_state=csf_state,
    mp2_amplitude=mp2_data,
    Norb=6
)
print(f"Orbital mapping: {mapping_dict}")

# Step 3: Extract W, V, N sets
S_W = [k for k, v in mapping_dict.items() if v == 'W']
S_V = [k for k, v in mapping_dict.items() if v == 'V']
S_N = [k for k, v in mapping_dict.items() if v == 'N']

# Step 4: Factorize the quantum state
factorization = factorize_state(
    basis_csr_state=quantum_state,
    S_w=S_W,
    S_v=S_V,
    S_n=S_N,
    state_type=state_type
)

# Result: Dictionary with tensor product factors
# factorization = {
#     (0, 1): psi_W,      # W block state
#     (2, 3): psi_V,      # V block state
#     (4,): psi_N1,       # N block orbital 4
#     (5,): psi_N2        # N block orbital 5
# }
```

---

## Implementation Notes

### Performance Considerations
- `partial_trace_einsum()` limited to ~26 qubits due to Einstein summation indexing
- Sparse matrix operations used for large state spaces
- Factorization reduces computational complexity by separating independent subsystems

### Numerical Precision
- Purity threshold: 1e-10 (distinguishes pure from mixed states)
- Coefficient matching threshold: 1e-6 (identifies state patterns)
- Normalization applied to prevent numerical drift

### Edge Cases
- Empty S_W, S_V, or S_N blocks handled gracefully
- Non-factorizable states return full-block representations
- Unknown state types return type 0 with None indices
