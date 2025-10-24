from utils_m2_factorize import get_indices_mapping_2_wvn
import sys
import pickle
from utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from utils_states import compress_state, convert_dense_format_to_sparse_format, convert_TZ_format_to_sparse_format

if __name__ == "__main__":
  bond_length     = sys.argv[1]
  filename        = f'h2o_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'

  with open(filename, 'rb') as f:
      (
      list_list_refCSF, # Vu |HF>
      list_list_Uext_mp2_CSF, # Wu Vu HF
      list_list_Uext_mp2_ampld, # Wu (i, a) pairs and the MP2 amplitudes
      list_list_Uext_opt_ampld, # Wu (i, a) with optimal amplitudes
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

  Nqubits = obt.shape[0]
  Norb    = Nqubits // 2
  dim     = 2 ** Nqubits
  tz_states       = []
  
  for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
     for j, ucsf in enumerate(ucsf_list):
        print(f"Basis state: {list_list_refCSF[i][j]}")
        tz_states.append(ucsf)
        index_mapping, state_type = get_indices_mapping_2_wvn(list_list_refCSF[i][j], list_list_Uext_mp2_ampld[i], Norb)
        print(f'CSF index mapping for bond length {bond_length}, CSF {len(tz_states)}: {index_mapping}')
        print(f'State type: {state_type}')
        # print(f"W amplitude: {list_list_Uext_mp2_ampld[i]})")
        # print(f"Vu CSF: {list_list_refCSF[i][j]})")



  statevectors = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in tz_states]
  tapered_state = convert_dense_format_to_sparse_format(compress_state(statevectors[0].toarray()[0]))
  print(f"Shape of tapered statevector: {type(tapered_state)}")